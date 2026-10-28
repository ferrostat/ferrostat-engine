//! Principal component analysis and factor analysis: `pca`, `pcamat`,
//! `factor`, `factormat`, `rotate`, their `estat` subcommands, `predict`
//! (component and factor scores) and the plots `screeplot`, `loadingplot`
//! and `scoreplot`.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt;
use crate::graph::{finish, Graph, Layer, LayerKind};
use crate::matrix::Matrix;
use crate::mv_rotate;
use crate::mvutil::{self, hdr, sig, Boxed};
use crate::store::{Results, Store};
use crate::util;

fn is_mv(st: &Store) -> Option<String> {
    match st.e.macros.get("cmd").map(String::as_str) {
        Some("pca") => Some("pca".into()),
        Some("factor") => Some("factor".into()),
        _ => None,
    }
}

fn mat(e: &Results, name: &str) -> Result<Matrix, String> {
    e.matrices.get(name).cloned().ok_or_else(|| rc_err(301, format!("e({name}) not found")))
}

fn evec(e: &Results, name: &str) -> Vec<f64> {
    e.matrices.get(name).map(|m| m.data.clone()).unwrap_or_default()
}

/// Variable names of the analysis.
fn vnames(e: &Results) -> Vec<String> {
    e.matrices.get("C").map(|m| m.rownames.clone()).unwrap_or_default()
}

fn has_rot(e: &Results) -> bool {
    e.macros.contains_key("r_criterion")
}

// ---------------------------------------------------------------------------
// shared input: data or matrix
// ---------------------------------------------------------------------------

struct Input {
    names: Vec<String>,
    /// covariance (or correlation if only that is known)
    c: Matrix,
    is_corr_input: bool,
    n: f64,
    means: Vec<f64>,
    sds: Vec<f64>,
    sample: Option<Vec<bool>>,
    wtype: String,
    matname: String,
}

fn input(ds: &DataSet, st: &Store, cmd: &Command, from_mat: bool) -> Result<Input, String> {
    if from_mat {
        let (c, names) = mvutil::read_sym_matrix(st, cmd)?;
        let c = if cmd.has_opt("forcepsd", 5) { mvutil::force_psd(&c) } else { c };
        let n = util::opt_num(cmd, "n", 1)?.ok_or_else(|| rc_err(198, "option n() required"))?;
        let p = names.len();
        let is_corr = (0..p).all(|i| (c[(i, i)] - 1.0).abs() < 1e-8);
        let sds = mvutil::vec_opt(st, cmd, "sds", p)?.unwrap_or_else(|| if is_corr { vec![1.0; p] } else { (0..p).map(|i| c[(i, i)].sqrt()).collect() });
        let means = mvutil::vec_opt(st, cmd, "means", p)?.unwrap_or_else(|| vec![0.0; p]);
        let matname = cmd.main.split_whitespace().next().unwrap_or("").to_string();
        return Ok(Input { names, c, is_corr_input: is_corr, n, means, sds, sample: None, wtype: String::new(), matname });
    }
    let vars = crate::varlist::expand(ds, &cmd.main)?;
    if vars.len() < 2 {
        return Err(rc_err(102, "too few variables specified"));
    }
    let d = mvutil::load(ds, st, cmd, &vars, "fa")?;
    if d.nrows() == 0 {
        return Err(rc_err(2000, "no observations"));
    }
    let (means, cov) = mvutil::moments(&d);
    let sds: Vec<f64> = (0..vars.len()).map(|i| cov[(i, i)].sqrt()).collect();
    Ok(Input { names: vars, c: cov, is_corr_input: false, n: d.n, means, sds, sample: Some(d.mask.clone()), wtype: d.wtype.clone(), matname: String::new() })
}

fn rowvec(v: &[f64], names: &[String]) -> Matrix {
    Matrix::row_vector(v).with_names(vec!["r1".into()], names.to_vec())
}

// ---------------------------------------------------------------------------
// pca / pcamat
// ---------------------------------------------------------------------------

pub struct Pca {
    pub mat: bool,
}

impl CommandHandler for Pca {
    fn name(&self) -> &'static str {
        if self.mat { "pcamat" } else { "pca" }
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if !self.mat && cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some("pca") {
                return Err(rc_err(301, "last estimates not found"));
            }
            return Ok(pca_display(st, cmd));
        }
        let inp = input(ds, st, cmd, self.mat)?;
        let p = inp.names.len();
        let covariance = cmd.has_opt("covariance", 3);
        if covariance && inp.is_corr_input && self.mat && cmd.opt("sds", 3).is_none() {
            // analyzing a correlation matrix as covariance is allowed
        }
        let c = if covariance { inp.c.clone() } else { mvutil::cov_to_corr(&inp.c) };
        let c = c.with_names(inp.names.clone(), inp.names.clone());
        let (vals, vecs) = mvutil::sym_eig(&c);
        let mut f = p;
        if let Some(k) = mvutil::opt_usize(cmd, "components", 3)?.or(mvutil::opt_usize(cmd, "factors", 3)?) {
            f = f.min(k.max(1));
        }
        if let Some(me) = util::opt_num(cmd, "mineigen", 3)? {
            f = f.min(vals.iter().filter(|&&v| v >= me).count().max(1));
        }
        let trace: f64 = vals.iter().sum();
        let l = mvutil::left_cols(&vecs, f).with_names(inp.names.clone(), mvutil::names("Comp", f));
        let psi: Vec<f64> = (0..p).map(|i| (f..p).map(|j| vals[j] * vecs[(i, j)] * vecs[(i, j)]).sum::<f64>().max(0.0)).map(|x| if x < 1e-12 * trace { 0.0 } else { x }).collect();
        let rho: f64 = vals[..f].iter().sum::<f64>() / trace;
        let n = inp.n;
        let comps = mvutil::names("Comp", p);
        let vce_normal = cmd.opt("vce", 3).map(|s| s.trim().to_lowercase().starts_with('n')).unwrap_or(false);
        let e = st.begin_e();
        e.set("N", n);
        e.set("f", f as f64);
        e.set("rho", rho);
        e.set("trace", trace);
        e.set("lndet", mvutil::ln_det(&c));
        e.set("cond", (vals[0] / vals[p - 1]).sqrt());
        e.set_str("cmd", "pca");
        e.set_str("cmdline", format!("{} {}", if self.mat { "pcamat" } else { "pca" }, cmd.raw.split_once(char::is_whitespace).map(|x| x.1).unwrap_or("").trim()));
        e.set_str("Ctype", if covariance { "covariance" } else { "correlation" });
        e.set_str("title", "Principal components");
        e.set_str("properties", "nob noV eigen");
        e.set_str("rotate_cmd", "pca_rotate");
        e.set_str("estat_cmd", "pca_estat");
        e.set_str("predict", "pca_p");
        e.set_str("marginsnotok", "_ALL");
        if !inp.wtype.is_empty() {
            e.set_str("wtype", inp.wtype.clone());
        }
        if self.mat {
            e.set_str("matrixname", inp.matname.clone());
        }
        e.set_mat("C", c.clone());
        e.set_mat("means", rowvec(&inp.means, &inp.names));
        e.set_mat("sds", rowvec(&inp.sds, &inp.names));
        e.set_mat("Ev", rowvec(&vals, &comps).with_names(vec!["Eigenvalues".into()], comps.clone()));
        e.set_mat("L", l.clone());
        e.set_mat("Psi", rowvec(&psi, &inp.names).with_names(vec!["Unexplained".into()], inp.names.clone()));
        let mut cm = Matrix::zeros(p, 4);
        let mut cum = 0.0;
        for j in 0..p {
            cum += vals[j];
            cm[(j, 0)] = vals[j];
            cm[(j, 1)] = if j + 1 < p { vals[j] - vals[j + 1] } else { SYSMISS };
            cm[(j, 2)] = vals[j] / trace;
            cm[(j, 3)] = cum / trace;
        }
        e.set_mat("components", cm.with_names(comps.clone(), vec!["Eigenvalue".into(), "Difference".into(), "Proportion".into(), "Cumulative".into()]));
        e.sample = inp.sample.clone();
        if vce_normal {
            pca_vce(st, &vals, &vecs, f, n, &inp.names, covariance);
        }
        Ok(pca_display(st, cmd))
    }
}

/// Asymptotic covariance of eigenvalues and retained eigenvectors under
/// multivariate normality.
fn pca_vce(st: &mut Store, vals: &[f64], vecs: &Matrix, f: usize, n: f64, names: &[String], _cov: bool) {
    let p = vals.len();
    let k = p + f * p;
    let mut b = vec![0.0; k];
    let mut bn = Vec::with_capacity(k);
    for j in 0..p {
        b[j] = vals[j];
        bn.push(format!("Eigenvalues:Comp{}", j + 1));
    }
    for c in 0..f {
        for i in 0..p {
            b[p + c * p + i] = vecs[(i, c)];
            bn.push(format!("Comp{}:{}", c + 1, names[i]));
        }
    }
    let mut v = Matrix::zeros(k, k);
    for i in 0..p {
        v[(i, i)] = 2.0 * vals[i] * vals[i] / n;
    }
    for a in 0..f {
        for c in 0..f {
            for r in 0..p {
                for s in 0..p {
                    let val = if a == c {
                        (0..p).filter(|&j| j != a).map(|j| vals[a] * vals[j] / (vals[a] - vals[j]).powi(2) * vecs[(r, j)] * vecs[(s, j)]).sum::<f64>() / n
                    } else {
                        -vals[a] * vals[c] / (vals[a] - vals[c]).powi(2) * vecs[(r, c)] * vecs[(s, a)] / n
                    };
                    v[(p + a * p + r, p + c * p + s)] = val;
                }
            }
        }
    }
    let trace: f64 = vals.iter().sum();
    let mut stats = Matrix::zeros(p, 5);
    let mut se = Matrix::zeros(p, 2);
    let mut bias = vec![0.0; p];
    let s2all: f64 = vals.iter().map(|x| x * x).sum();
    for i in 0..p {
        bias[i] = vals[i] / n * (0..p).filter(|&j| j != i).map(|j| vals[j] / (vals[i] - vals[j])).sum::<f64>();
        let pi = vals[i] / trace;
        let vp = 2.0 / n * ((1.0 - pi).powi(2) * vals[i] * vals[i] + pi * pi * (s2all - vals[i] * vals[i])) / (trace * trace);
        let psi: f64 = vals[..=i].iter().sum::<f64>() / trace;
        let s_in: f64 = vals[..=i].iter().map(|x| x * x).sum();
        let vc = 2.0 / n * ((1.0 - psi).powi(2) * s_in + psi * psi * (s2all - s_in)) / (trace * trace);
        stats[(i, 0)] = vals[i];
        stats[(i, 1)] = pi;
        stats[(i, 2)] = psi;
        stats[(i, 3)] = bias[i];
        stats[(i, 4)] = vals[i] + 0.0;
        se[(i, 0)] = vp.max(0.0).sqrt();
        se[(i, 1)] = if i + 1 == p { 0.0 } else { vc.max(0.0).sqrt() };
    }
    let rank = k as f64;
    let c = st.e.matrices.get("C").cloned().unwrap();
    let ldet = mvutil::ln_det(&c);
    let pf = p as f64;
    let chi2_i = -(n - (2.0 * pf + 5.0) / 6.0) * ldet;
    let df_i = pf * (pf - 1.0) / 2.0;
    let ce = mvutil::cov_to_corr(&c);
    let (lt, _) = mvutil::sym_eig(&ce);
    let chi2_s = -(n - (2.0 * pf * pf + pf + 2.0) / (6.0 * pf)) * (lt.iter().map(|x| x.ln()).sum::<f64>() - pf * (lt.iter().sum::<f64>() / pf).ln());
    let df_s = (pf + 2.0) * (pf - 1.0) / 2.0;
    let e = &mut st.e;
    let comps = mvutil::names("Comp", p);
    e.set_mat("b", Matrix::row_vector(&b).with_names(vec!["y1".into()], bn.clone()));
    e.set_mat("V", v.with_names(bn.clone(), bn));
    e.set("rank", rank);
    e.set("v_rho", se[(f - 1, 1)].powi(2));
    e.set("chi2_i", chi2_i);
    e.set("df_i", df_i);
    e.set("p_i", dist::chi2tail(df_i, chi2_i));
    e.set("chi2_s", chi2_s);
    e.set("df_s", df_s);
    e.set("p_s", dist::chi2tail(df_s, chi2_s));
    e.set_str("vce", "multivariate normality");
    e.set_str("properties", "b V eigen");
    e.set_mat("Ev_bias", Matrix::row_vector(&bias).with_names(vec!["bias".into()], comps.clone()));
    e.set_mat("Ev_stats", stats.with_names(comps.clone(), vec!["Eigenvalue".into(), "Proportion".into(), "Cumulative".into(), "Bias".into(), "Ev".into()]));
    e.set_mat("Ev_se", se.with_names(comps, vec!["SE_Prop".into(), "SE_Cum".into()]));
}

fn ev_table_rows(stub: &str, names: &[String], vals: &[f64], col1: &str, fmtv: &dyn Fn(f64) -> String, divisor: f64, oblique: Option<&str>) -> Vec<String> {
    let mut out = Vec::new();
    let rule = format!("    {}", "-".repeat(74));
    out.push(rule.clone());
    if let Some(note) = oblique {
        out.push(format!("    {:>12} | {:>12} {:>12}    {}", stub, col1, "Proportion", note));
    } else {
        out.push(format!("    {:>12} | {:>12} {:>12} {:>18} {:>12}", stub, col1, "Difference", "Proportion", "Cumulative"));
    }
    out.push(format!("    {}+{}", "-".repeat(13), "-".repeat(60)));
    let mut cum = 0.0;
    for (j, v) in vals.iter().enumerate() {
        cum += v;
        if oblique.is_some() {
            out.push(format!("    {:>12} | {:>12} {:>12}", names[j], fmtv(*v), format!("{:.4}", v / divisor)));
        } else {
            let diff = if j + 1 < vals.len() { fmtv(v - vals[j + 1]) } else { ".".into() };
            out.push(format!("    {:>12} | {:>12} {:>12} {:>18} {:>12}", names[j], fmtv(*v), diff, format!("{:.4}", v / divisor), format!("{:.4}", cum / divisor)));
        }
    }
    out.push(rule);
    out
}

/// Loadings table with blocks of columns; `extra` is the trailing column
/// (Unexplained / Uniqueness) shown in the last block.
fn loading_table(names: &[String], colnames: &[String], l: &Matrix, cell: &dyn Fn(f64) -> String, extra: Option<(&str, Vec<String>)>, colw: usize) -> Vec<String> {
    let k = l.cols;
    let mut out = Vec::new();
    let extra_w = extra.as_ref().map(|(h, _)| h.len().max(10) + 2).unwrap_or(0);
    let maxcols = ((80 - 4 - 14 - 1) / colw).max(1);
    let mut c0 = 0;
    loop {
        let remaining = k - c0;
        let fits_with_extra = remaining * colw + if extra.is_some() { extra_w + 2 } else { 0 } <= 80 - 4 - 14 - 1;
        let c1 = if fits_with_extra { k } else { (c0 + maxcols).min(k) };
        let last = c1 == k && (fits_with_extra || extra.is_none());
        let mut groups = vec![colnames[c0..c1].iter().map(|c| (c.clone(), colw)).collect::<Vec<_>>()];
        if last {
            if let Some((h, _)) = &extra {
                groups.push(vec![(h.to_string(), extra_w)]);
            }
        }
        if groups[0].is_empty() {
            groups.remove(0);
        }
        let mut b = Boxed::new("Variable", groups);
        for (i, nm) in names.iter().enumerate() {
            let mut cells: Vec<String> = (c0..c1).map(|j| cell(l[(i, j)])).collect();
            if last {
                if let Some((_, ex)) = &extra {
                    cells.push(ex[i].clone());
                }
            }
            b.row(nm, cells);
        }
        if c0 > 0 {
            out.push(String::new());
        }
        out.extend(b.render());
        c0 = c1;
        if last {
            break;
        }
        if c0 >= k && extra.is_some() {
            // only the extra column remains
            let mut b = Boxed::new("Variable", vec![vec![(extra.as_ref().unwrap().0.to_string(), extra_w)]]);
            for (i, nm) in names.iter().enumerate() {
                b.row(nm, vec![extra.as_ref().unwrap().1[i].clone()]);
            }
            out.push(String::new());
            out.extend(b.render());
            break;
        }
    }
    out
}

fn blank_cell(blanks: f64, fmt_: &'static str) -> impl Fn(f64) -> String {
    move |v: f64| if v.abs() < blanks { String::new() } else { fmt(v, fmt_).trim().to_string() }
}

fn pca_display(st: &Store, cmd: &Command) -> String {
    let e = &st.e;
    let names = vnames(e);
    let p = names.len();
    let vals = evec(e, "Ev");
    let f = e.get("f").unwrap_or(p as f64) as usize;
    let trace = e.get("trace").unwrap_or(0.0);
    let ctype = e.macros.get("Ctype").cloned().unwrap_or_default();
    let rotated = has_rot(e) && !cmd.has_opt("norotated", 5);
    let n = e.get("N").unwrap_or(0.0);
    let blanks = util::opt_num(cmd, "blanks", 3).ok().flatten().unwrap_or(0.0);
    let vce = e.matrices.contains_key("V") && !cmd.has_opt("novce", 5);
    let mut out = vec![String::new()];
    if vce && !rotated {
        if ctype == "correlation" {
            out.push("(with PCA/correlation, SEs and tests are approximate)".into());
            out.push(String::new());
        }
        out.push(hdr(&format!("Principal components/{ctype}"), "Number of obs", &util::commas(n as usize)));
        out.push(hdr("", "Number of comp.", &f.to_string()));
        out.push(hdr("", "Trace", &sig(trace, 8)));
        out.push(hdr("", "Rho", &format!("{:.4}", e.get("rho").unwrap_or(0.0))));
        out.push(hdr("SEs assume multivariate normality", "SE(Rho)", &format!("{:.4}", e.get("v_rho").unwrap_or(0.0).sqrt())));
        out.push(String::new());
        let b = mat(e, "b").unwrap();
        let v = mat(e, "V").unwrap();
        let level = util::level(cmd, st).unwrap_or(95.0);
        let k = p + f * p;
        let bn: Vec<String> = b.colnames.iter().take(k).cloned().collect();
        out.extend(crate::est::coef_table(&crate::est::TableSpec { depvar: "", names: &bn, b: &b.data[..k], v: &v, omitted: &vec![false; k], is_base: &vec![false; k], df: None, level, eform: None, coef_title: "Coefficient" }));
        out.push(String::new());
        out.push(format!("LR test for independence:  chi2({}) = {:>9.2}   Prob > chi2 = {:.4}", e.get("df_i").unwrap_or(0.0), e.get("chi2_i").unwrap_or(0.0), e.get("p_i").unwrap_or(0.0)));
        out.push(format!("LR test for   sphericity:  chi2({}) = {:>9.2}   Prob > chi2 = {:.4}", e.get("df_s").unwrap_or(0.0), e.get("chi2_s").unwrap_or(0.0), e.get("p_s").unwrap_or(0.0)));
        out.push(String::new());
        out.push("Explained variance by components".into());
        out.push(String::new());
        let stats = mat(e, "Ev_stats").unwrap();
        let se = mat(e, "Ev_se").unwrap();
        out.push(format!("  {}", "-".repeat(76)));
        out.push(format!("  {:>12} | {:>12} {:>12} {:>10} {:>12} {:>10} {:>12}", "Components", "Eigenvalue", "Proportion", "SE_Prop", "Cumulative", "SE_Cum", "Bias"));
        out.push(format!("  {}+{}", "-".repeat(13), "-".repeat(62)));
        for j in 0..p {
            out.push(format!("  {:>12} | {:>12} {:>12} {:>10} {:>12} {:>10} {:>12}", format!("Comp{}", j + 1), sig(vals[j], 7), format!("{:.4}", stats[(j, 1)]), format!("{:.4}", se[(j, 0)]), format!("{:.4}", stats[(j, 2)]), format!("{:.4}", se[(j, 1)]), sig(stats[(j, 3)], 5)));
        }
        out.push(format!("  {}", "-".repeat(76)));
        return out.join("\n");
    }
    out.push(hdr(&format!("Principal components/{ctype}"), "Number of obs", &util::commas(n as usize)));
    out.push(hdr("", "Number of comp.", &f.to_string()));
    out.push(hdr("", "Trace", &sig(trace, 8)));
    let rtext = if rotated { mv_rotate::rotation_text(e) } else { "(unrotated = principal)".into() };
    out.push(hdr(&format!("    Rotation: {rtext}"), "Rho", &format!("{:.4}", e.get("rho").unwrap_or(0.0))));
    out.push(String::new());
    let comps = mvutil::names("Comp", p);
    if rotated {
        let rev = evec(e, "r_Ev");
        let obl = e.macros.get("r_class").map(String::as_str) == Some("oblique");
        out.extend(ev_table_rows("Component", &comps, &rev, "Variance", &|v| sig(v, 6), trace, if obl { Some("Rotated comp. are correlated") } else { None }));
    } else {
        out.extend(ev_table_rows("Component", &comps, &vals, "Eigenvalue", &|v| sig(v, 6), trace, None));
    }
    out.push(String::new());
    let l = if rotated { mat(e, "r_L").unwrap() } else { mat(e, "L").unwrap() };
    let psi = evec(e, "Psi");
    out.push(if rotated { "Rotated components ".into() } else { "Principal components (eigenvectors) ".into() });
    out.push(String::new());
    let cell = blank_cell(blanks, "%8.4f");
    out.extend(loading_table(&names, &comps[..f], &l, &cell, Some(("Unexplained", psi.iter().map(|x| sig(*x, 4)).collect())), 10));
    if blanks > 0.0 {
        out.push(format!("    (blanks are abs(loading)<{})", sig(blanks, 8)));
    }
    if rotated {
        let t = mat(e, "r_T").unwrap();
        out.push(String::new());
        out.push("Component rotation matrix".into());
        out.push(String::new());
        out.extend(rot_matrix_table(&t, "Comp"));
    }
    out.join("\n")
}

fn rot_matrix_table(t: &Matrix, prefix: &str) -> Vec<String> {
    let k = t.cols;
    let cn = mvutil::names(prefix, k);
    let mut b = Boxed::new("", vec![cn.iter().map(|c| (c.clone(), 10)).collect()]);
    for i in 0..k {
        b.row(&cn[i], (0..k).map(|j| format!("{:.4}", t[(i, j)])).collect());
    }
    b.render()
}

// ---------------------------------------------------------------------------
// factor / factormat
// ---------------------------------------------------------------------------

pub struct Factor {
    pub mat: bool,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Method {
    Pf,
    Pcf,
    Ipf,
    Ml,
}

impl Method {
    fn code(self) -> &'static str {
        match self {
            Method::Pf => "pf",
            Method::Pcf => "pcf",
            Method::Ipf => "ipf",
            Method::Ml => "ml",
        }
    }
    fn title(self) -> &'static str {
        match self {
            Method::Pf => "principal factors",
            Method::Pcf => "principal-component factors",
            Method::Ipf => "iterated principal factors",
            Method::Ml => "maximum likelihood",
        }
    }
}

fn smc_of(r: &Matrix) -> Vec<f64> {
    let ri = mvutil::sym_inv(r);
    (0..r.rows).map(|i| 1.0 - 1.0 / ri[(i, i)]).collect()
}

/// Eigen decomposition of R with the diagonal replaced by `h`.
fn reduced_eig(r: &Matrix, h: &[f64]) -> (Vec<f64>, Matrix) {
    let mut rr = r.clone();
    for i in 0..r.rows {
        rr[(i, i)] = h[i];
    }
    mvutil::sym_eig(&rr)
}

fn loadings_from(vals: &[f64], vecs: &Matrix, q: usize) -> Matrix {
    let p = vecs.rows;
    let mut l = Matrix::zeros(p, q);
    for j in 0..q {
        let s = vals[j].max(0.0).sqrt();
        for i in 0..p {
            l[(i, j)] = vecs[(i, j)] * s;
        }
    }
    l
}

/// Maximum-likelihood factor solution for `q` factors; returns (psi, loadings,
/// minimized discrepancy F, iteration log of F).
fn ml_fit(r: &Matrix, q: usize, start: Option<Vec<f64>>) -> (Vec<f64>, Matrix, f64, Vec<f64>) {
    let p = r.rows;
    let lower = 0.005;
    let fval = |psi: &[f64]| -> (f64, Vec<f64>, Matrix) {
        let mut ss = r.clone();
        for i in 0..p {
            for j in 0..p {
                ss[(i, j)] = r[(i, j)] / (psi[i] * psi[j]).sqrt();
            }
        }
        let (g, v) = mvutil::sym_eig(&ss);
        let f: f64 = g[q..].iter().map(|x| x - x.max(1e-300).ln() - 1.0).sum();
        (f, g, v)
    };
    let lam_of = |psi: &[f64], g: &[f64], v: &Matrix| -> Matrix {
        let mut l = Matrix::zeros(p, q);
        for j in 0..q {
            let s = (g[j] - 1.0).max(0.0).sqrt();
            for i in 0..p {
                l[(i, j)] = psi[i].sqrt() * v[(i, j)] * s;
            }
        }
        l
    };
    let mut psi = start.unwrap_or_else(|| smc_of(r).iter().map(|s| (1.0 - s).max(lower)).collect());
    let (mut f, mut g, mut v) = fval(&psi);
    let mut log = vec![f];
    for _ in 0..500 {
        let lam = lam_of(&psi, &g, &v);
        let mut sig = lam.mul(&lam.transpose()).unwrap();
        for i in 0..p {
            sig[(i, i)] += psi[i];
        }
        let si = mvutil::sym_inv(&sig);
        let d = sig.sub(r).unwrap();
        let m = si.mul(&d).unwrap().mul(&si).unwrap();
        let grad: Vec<f64> = (0..p).map(|i| m[(i, i)]).collect();
        let mut h = Matrix::zeros(p, p);
        for i in 0..p {
            for j in 0..p {
                h[(i, j)] = si[(i, j)] * si[(i, j)];
            }
        }
        // Newton step with a finite-difference Hessian of the concentrated criterion
        let grad_at = |ps: &[f64]| -> Vec<f64> {
            let (_, g2, v2) = fval(ps);
            let lam = lam_of(ps, &g2, &v2);
            let mut sig = lam.mul(&lam.transpose()).unwrap();
            for i in 0..p {
                sig[(i, i)] += ps[i];
            }
            let si = mvutil::sym_inv(&sig);
            let m = si.mul(&sig.sub(r).unwrap()).unwrap().mul(&si).unwrap();
            (0..p).map(|i| m[(i, i)]).collect()
        };
        {
            let mut hn = Matrix::zeros(p, p);
            for j in 0..p {
                let hstep = 1e-6 * psi[j].max(1e-3);
                let mut pp = psi.clone();
                pp[j] += hstep;
                let gp = grad_at(&pp);
                let mut pm = psi.clone();
                pm[j] -= hstep;
                let gm = grad_at(&pm);
                for i in 0..p {
                    hn[(i, j)] = (gp[i] - gm[i]) / (2.0 * hstep);
                }
            }
            let mut sym = hn.clone();
            for i in 0..p {
                for j in 0..p {
                    sym[(i, j)] = 0.5 * (hn[(i, j)] + hn[(j, i)]);
                }
            }
            if sym.cholesky().is_ok() {
                h = sym;
            }
        }
        // free parameters: not stuck at the bound with a positive gradient
        let free: Vec<usize> = (0..p).filter(|&i| !(psi[i] <= lower + 1e-12 && grad[i] > 0.0)).collect();
        let mut step = vec![0.0; p];
        if !free.is_empty() {
            let mut hs = Matrix::zeros(free.len(), free.len());
            for (a, &i) in free.iter().enumerate() {
                for (b, &j) in free.iter().enumerate() {
                    hs[(a, b)] = h[(i, j)];
                }
            }
            let hi = mvutil::sym_inv(&hs);
            for (a, &i) in free.iter().enumerate() {
                step[i] = -(0..free.len()).map(|b| hi[(a, b)] * grad[free[b]]).sum::<f64>();
            }
        }
        let mut t = 1.0;
        let mut improved = false;
        for _ in 0..40 {
            let cand: Vec<f64> = (0..p).map(|i| (psi[i] + t * step[i]).max(lower)).collect();
            let (fc, gc, vc) = fval(&cand);
            if fc <= f + 1e-15 {
                let df = f - fc;
                psi = cand;
                f = fc;
                g = gc;
                v = vc;
                improved = true;
                log.push(f);
                if df < 1e-12 {
                    let lam = lam_of(&psi, &g, &v);
                    return (psi, lam, f, log);
                }
                break;
            }
            t /= 2.0;
        }
        if !improved {
            break;
        }
    }
    let lam = lam_of(&psi, &g, &v);
    (psi, lam, f, log)
}

fn n_params(p: usize, q: usize) -> usize {
    (p * q).saturating_sub(q * q.saturating_sub(1) / 2).min(p * p.saturating_sub(1) / 2)
}

impl CommandHandler for Factor {
    fn name(&self) -> &'static str {
        if self.mat { "factormat" } else { "factor" }
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if !self.mat && cmd.main.trim().is_empty() {
            if st.e.macros.get("cmd").map(String::as_str) != Some("factor") {
                return Err(rc_err(301, "last estimates not found"));
            }
            return Ok(factor_display(st, cmd));
        }
        let inp = input(ds, st, cmd, self.mat)?;
        let p = inp.names.len();
        let r = mvutil::cov_to_corr(&inp.c).with_names(inp.names.clone(), inp.names.clone());
        let method = if cmd.has_opt("ml", 2) {
            Method::Ml
        } else if cmd.has_opt("ipf", 3) {
            Method::Ipf
        } else if cmd.has_opt("pcf", 3) {
            Method::Pcf
        } else {
            Method::Pf
        };
        let maxf = mvutil::opt_usize(cmd, "factors", 3)?;
        let mineig = util::opt_num(cmd, "mineigen", 3)?.map(|x| x.max(5e-6)).unwrap_or(if method == Method::Pcf { 1.0 } else { 5e-6 });
        let n = inp.n;
        let pf = p as f64;
        let ldet = mvutil::ln_det(&r);
        let chi2_i = -(n - (2.0 * pf + 5.0) / 6.0) * ldet;
        let df_i = pf * (pf - 1.0) / 2.0;
        let quiet_log = cmd.has_opt("nolog", 5);
        let mut pre = vec![format!("(obs={})", util::commas(n as usize))];
        let (vals, l, psi, q): (Vec<f64>, Matrix, Vec<f64>, usize);
        let mut ml_info: Option<(f64, f64, Vec<f64>)> = None;
        match method {
            Method::Pf | Method::Pcf => {
                let h = if method == Method::Pf { smc_of(&r) } else { vec![1.0; p] };
                let (v, vecs) = reduced_eig(&r, &h);
                let mut k = v.iter().filter(|&&x| x > mineig).count();
                if let Some(m) = maxf {
                    k = k.min(m);
                }
                let lq = loadings_from(&v, &vecs, k);
                psi = (0..p).map(|i| 1.0 - (0..k).map(|j| lq[(i, j)].powi(2)).sum::<f64>()).collect();
                vals = v;
                l = lq;
                q = k;
            }
            Method::Ipf => {
                let citer = mvutil::opt_usize(cmd, "citerate", 2)?;
                let mut h = smc_of(&r);
                let mut it = 0;
                let (mut v, mut vecs) = reduced_eig(&r, &h);
                loop {
                    let mut k = v.iter().filter(|&&x| x > mineig).count();
                    if let Some(m) = maxf {
                        k = k.min(m);
                    }
                    let lq = loadings_from(&v, &vecs, k);
                    let hn: Vec<f64> = (0..p).map(|i| (0..k).map(|j| lq[(i, j)].powi(2)).sum::<f64>()).collect();
                    let ch = hn.iter().zip(&h).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
                    h = hn;
                    it += 1;
                    let (v2, vecs2) = reduced_eig(&r, &h);
                    v = v2;
                    vecs = vecs2;
                    let done = match citer {
                        Some(c) => it >= c,
                        None => ch < 1e-4 || it >= 1000,
                    };
                    if done {
                        break;
                    }
                }
                let mut k = v.iter().filter(|&&x| x > mineig).count();
                if let Some(m) = maxf {
                    k = k.min(m);
                }
                let lq = loadings_from(&v, &vecs, k);
                psi = (0..p).map(|i| 1.0 - (0..k).map(|j| lq[(i, j)].powi(2)).sum::<f64>()).collect();
                vals = v;
                l = lq;
                q = k;
            }
            Method::Ml => {
                if p < 3 {
                    return Err(rc_err(198, "at least three variables required with ml"));
                }
                let maxq = (1..=p).take_while(|&q| p * q - q * (q - 1) / 2 <= p * (p - 1) / 2).last().unwrap_or(1);
                let mut k = maxq.max(1);
                if let Some(m) = maxf {
                    k = k.min(m.max(1));
                }
                let (ps, lam, fmin, log) = ml_fit(&r, k, None);
                if !quiet_log {
                    for (i, fv) in log.iter().enumerate() {
                        pre.push(format!("Iteration {i}:   Log likelihood = {}", sig(-n / 2.0 * fv, 8)));
                    }
                }
                let mut lam = lam;
                mvutil::orient_columns(&mut lam);
                let ev: Vec<f64> = (0..k).map(|j| (0..p).map(|i| lam[(i, j)].powi(2)).sum()).collect();
                // order factors by explained variance
                let mut ord: Vec<usize> = (0..k).collect();
                ord.sort_by(|&a, &b| ev[b].total_cmp(&ev[a]));
                let mut l2 = Matrix::zeros(p, k);
                for (nj, &oj) in ord.iter().enumerate() {
                    for i in 0..p {
                        l2[(i, nj)] = lam[(i, oj)];
                    }
                }
                vals = ord.iter().map(|&j| ev[j]).collect();
                psi = ps;
                l = l2;
                q = k;
                ml_info = Some((fmin, -n / 2.0 * fmin, log));
            }
        }
        let names = inp.names.clone();
        let fnames = mvutil::names("Factor", q);
        let df_m = n_params(p, q) as f64;
        let evsum: f64 = vals.iter().sum();
        let heywood = psi.iter().any(|&x| x <= 0.0 || (method == Method::Ml && x <= 0.0051));
        let e = st.begin_e();
        e.set("N", n);
        e.set("f", q as f64);
        e.set("evsum", evsum);
        e.set("df_m", df_m);
        e.set("df_r", df_i - df_m);
        e.set("chi2_i", chi2_i);
        e.set("df_i", df_i);
        e.set("p_i", dist::chi2tail(df_i, chi2_i));
        e.set_str("cmd", "factor");
        e.set_str("cmdline", format!("{} {}", if self.mat { "factormat" } else { "factor" }, cmd.raw.split_once(char::is_whitespace).map(|x| x.1).unwrap_or("").trim()));
        e.set_str("method", method.code());
        e.set_str("title", "Factor analysis");
        e.set_str("mtitle", method.title());
        e.set_str("properties", "nob noV eigen");
        e.set_str("rotate_cmd", "factor_rotate");
        e.set_str("estat_cmd", "factor_estat");
        e.set_str("predict", "factor_p");
        e.set_str("marginsnotok", "_ALL");
        if let Some(m) = maxf {
            e.set_str("factors", format!("factors({m})"));
        }
        if let Some(m) = util::opt_num(cmd, "mineigen", 3)? {
            e.set_str("mineigen", format!("mineigen({})", crate::format::fmt_g(m, 9)));
        }
        if heywood {
            e.set_str("heywood", "Heywood case");
        }
        if !inp.wtype.is_empty() {
            e.set_str("wtype", inp.wtype.clone());
        }
        if self.mat {
            e.set_str("matrixname", inp.matname.clone());
        }
        e.set_mat("C", r.clone());
        e.set_mat("means", rowvec(&inp.means, &names));
        e.set_mat("sds", rowvec(&inp.sds, &names));
        e.set_mat("L", l.clone().with_names(names.clone(), fnames.clone()));
        e.set_mat("Psi", rowvec(&psi, &names).with_names(vec!["psi".into()], names.clone()));
        e.set_mat("Phi", Matrix::identity(q).with_names(fnames.clone(), fnames.clone()));
        let evn = mvutil::names("Factor", vals.len());
        e.set_mat("Ev", rowvec(&vals, &evn).with_names(vec!["Ev".into()], evn));
        if let Some((fmin, ll, _)) = &ml_info {
            let k = df_m;
            e.set("ll", *ll);
            e.set("ll_0", n / 2.0 * ldet);
            e.set("aic", -2.0 * ll + 2.0 * k);
            e.set("bic", -2.0 * ll + k * n.ln());
            let df1 = df_i - df_m;
            let chi2_1 = (n - (2.0 * pf + 5.0) / 6.0 - 2.0 * q as f64 / 3.0) * fmin;
            e.set("chi2_1", chi2_1);
            e.set("df_1", df1);
            e.set("p_1", if df1 > 0.0 { dist::chi2tail(df1, chi2_1) } else { SYSMISS });
        }
        e.sample = inp.sample.clone();
        let mut out = pre;
        out.push(factor_display(st, cmd));
        Ok(out.join("\n"))
    }
}

fn factor_display(st: &Store, cmd: &Command) -> String {
    let e = &st.e;
    let names = vnames(e);
    let vals = evec(e, "Ev");
    let q = e.get("f").unwrap_or(0.0) as usize;
    let n = e.get("N").unwrap_or(0.0);
    let method = e.macros.get("method").cloned().unwrap_or_default();
    let rotated = has_rot(e) && !cmd.has_opt("norotated", 5);
    let blanks = util::opt_num(cmd, "blanks", 3).ok().flatten().unwrap_or(0.0);
    let mut out = vec![String::new()];
    out.push(hdr("Factor analysis/correlation", "Number of obs", &util::commas(n as usize)));
    out.push(hdr(&format!("    Method: {}", e.macros.get("mtitle").cloned().unwrap_or_default()), "Retained factors", &q.to_string()));
    let rtext = if rotated { mv_rotate::rotation_text(e) } else { "(unrotated)".into() };
    out.push(hdr(&format!("    Rotation: {rtext}"), "Number of params", &crate::format::fmt_g(e.get("df_m").unwrap_or(0.0), 10)));
    if method == "ml" {
        out.push(hdr("", "Schwarz's BIC", &sig(e.get("bic").unwrap_or(0.0), 6)));
        out.push(hdr(&format!("    Log likelihood = {}", sig(e.get("ll").unwrap_or(0.0), 7)), "(Akaike's) AIC", &sig(e.get("aic").unwrap_or(0.0), 6)));
    }
    if e.macros.contains_key("heywood") {
        out.push("    Beware: solution is a Heywood case".into());
        out.push("            (i.e., invalid or boundary values of uniqueness)".into());
    }
    out.push(String::new());
    let divisor = if cmd.has_opt("altdivisor", 3) { mat(e, "C").map(|c| c.trace()).unwrap_or(1.0) } else { vals.iter().sum::<f64>() };
    if rotated {
        let rev = evec(e, "r_Ev");
        let obl = e.macros.get("r_class").map(String::as_str) == Some("oblique");
        let fnames = mvutil::names("Factor", rev.len());
        let div = if obl { divisor } else { rev.iter().sum::<f64>() };
        out.extend(ev_table_rows("Factor", &fnames, &rev, "Variance", &|v| format!("{v:.5}"), div, if obl { Some("Rotated factors are correlated") } else { None }));
    } else {
        let fnames = mvutil::names("Factor", vals.len());
        out.extend(ev_table_rows("Factor", &fnames, &vals, "Eigenvalue", &|v| format!("{v:.5}"), divisor, None));
    }
    out.push(format!("    LR test: independent vs. saturated:  chi2({}) = {:>8.2} Prob>chi2 = {:.4}", e.get("df_i").unwrap_or(0.0), e.get("chi2_i").unwrap_or(0.0), e.get("p_i").unwrap_or(0.0)));
    if method == "ml" {
        let df1 = e.get("df_1").unwrap_or(0.0);
        let lab = format!("{:>3} factor{} vs. saturated", q, if q == 1 { " " } else { "s" });
        let pv = e.get("p_1").unwrap_or(SYSMISS);
        out.push(format!("    LR test: {lab}:  chi2({}) = {:>8.2} Prob>chi2 = {}", df1, e.get("chi2_1").unwrap_or(0.0), if is_missing(pv) { ".".to_string() } else { format!("{pv:.4}") }));
    }
    out.push(String::new());
    let l = if rotated { mat(e, "r_L").unwrap() } else { mat(e, "L").unwrap() };
    let psi = evec(e, "Psi");
    out.push(if rotated { "Rotated factor loadings (pattern matrix) and unique variances".into() } else { "Factor loadings (pattern matrix) and unique variances".into() });
    out.push(String::new());
    let fn_ = mvutil::names("Factor", q);
    let cell = blank_cell(blanks, "%8.4f");
    out.extend(loading_table(&names, &fn_, &l, &cell, Some(("Uniqueness", psi.iter().map(|x| format!("{x:.4}")).collect())), 10));
    if blanks > 0.0 {
        out.push(format!("    (blanks represent abs(loading)<{})", sig(blanks, 8)));
    }
    if rotated {
        out.push(String::new());
        out.push("Factor rotation matrix".into());
        out.push(String::new());
        out.extend(rot_matrix_table(&mat(e, "r_T").unwrap(), "Factor"));
    }
    out.join("\n")
}

// ---------------------------------------------------------------------------
// rotate
// ---------------------------------------------------------------------------

pub struct Rotate;

impl CommandHandler for Rotate {
    fn name(&self) -> &'static str {
        "rotate"
    }

    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let which = is_mv(st).ok_or_else(|| rc_err(321, "rotate is allowed only after pca, pcamat, factor, factormat"))?;
        if cmd.has_opt("clear", 5) {
            let e = &mut st.e;
            e.scalars.retain(|k, _| !k.starts_with("r_"));
            e.macros.retain(|k, _| !k.starts_with("r_"));
            e.matrices.retain(|k, _| !k.starts_with("r_"));
            return Ok(String::new());
        }
        let o = mv_rotate::parse_opts(st, cmd)?;
        let a = mat(&st.e, "L")?;
        let k = a.cols;
        let nf = mvutil::opt_usize(cmd, "factors", 1)?.or(mvutil::opt_usize(cmd, "components", 4)?).unwrap_or(k).min(k);
        let mut r = mv_rotate::rotate(&a, &o, nf)?;
        let c = mat(&st.e, "C")?;
        // explained variance by rotated factors / components
        let var_of = |r: &mv_rotate::RotResult| -> Vec<f64> {
            if which == "pca" {
                let m = r.l.transpose().mul(&c).unwrap().mul(&r.l).unwrap();
                m.diag()
            } else {
                // explained variance: column sums of squares of the structure matrix L Phi
                let s = r.l.mul(&r.phi).unwrap();
                (0..k).map(|j| (0..s.rows).map(|i| s[(i, j)].powi(2)).sum()).collect()
            }
        };
        let v0 = var_of(&r);
        let mut order: Vec<usize> = (0..nf).collect();
        order.sort_by(|&x, &y| v0[y].total_cmp(&v0[x]));
        order.extend(nf..k);
        mv_rotate::permute(&mut r, &order);
        let ev = var_of(&r);
        let pre = if which == "pca" { "Comp" } else { "Factor" };
        let names = vnames(&st.e);
        let cn = mvutil::names(pre, k);
        let e = &mut st.e;
        e.scalars.retain(|k, _| !k.starts_with("r_"));
        e.macros.retain(|k, _| !k.starts_with("r_"));
        e.matrices.retain(|k, _| !k.starts_with("r_"));
        e.set("r_f", nf as f64);
        e.set("r_fmin", r.fmin);
        e.set_str("r_class", r.class.clone());
        e.set_str("r_criterion", o.crit.name());
        e.set_str("r_ctitle", r.ctitle.clone());
        e.set_str("r_normalization", if o.normalize { "kaiser" } else { "none" });
        e.set_mat("r_L", r.l.clone().with_names(names, cn.clone()));
        e.set_mat("r_T", r.t.clone().with_names(cn.clone(), cn.clone()));
        e.set_mat("r_Ev", Matrix::row_vector(&ev).with_names(vec!["r_Ev".into()], cn.clone()));
        if which == "factor" {
            e.set_mat("r_Phi", r.phi.clone().with_names(cn.clone(), cn));
        }
        let mut c2 = cmd.clone();
        c2.opts.retain(|(n, _)| n.to_lowercase().starts_with("bl"));
        Ok(if which == "pca" { pca_display(st, &c2) } else { factor_display(st, &c2) })
    }
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------

/// Current loading matrix (rotated unless `norotated`), and Phi for factor.
fn current_loadings(e: &Results, cmd: &Command) -> (Matrix, Matrix, bool) {
    let rot = has_rot(e) && !cmd.has_opt("norotated", 5);
    let l = if rot { e.matrices.get("r_L").cloned().unwrap() } else { e.matrices.get("L").cloned().unwrap() };
    let k = l.cols;
    let phi = if rot { e.matrices.get("r_Phi").cloned().unwrap_or_else(|| Matrix::identity(k)) } else { Matrix::identity(k) };
    (l, phi, rot)
}

pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<Option<String>, String> {
    let Some(which) = is_mv(st) else { return Ok(None) };
    let e = st.e.clone();
    let names = vnames(&e);
    let p = names.len();
    let (l, phi, _rot) = current_loadings(&e, cmd);
    let f = l.cols;
    let means = evec(&e, "means");
    let sds = evec(&e, "sds");
    let ctype = e.macros.get("Ctype").cloned().unwrap_or_else(|| "correlation".into());
    let corr = which == "factor" || ctype == "correlation";
    let mut mask = util::touse(ds, st, cmd)?;
    for v in &names {
        if ds.get_var(v).is_none() {
            return Err(rc_err(111, format!("variable {v} not found")));
        }
    }
    util::markout(ds, &mut mask, &names);
    let rows = util::rows(&mask);
    let cols: Vec<Vec<f64>> = names.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let center = corr || cmd.has_opt("center", 3);
    let z = |i: usize, t: usize| -> f64 {
        let x = cols[i][t];
        if corr {
            (x - means[i]) / sds[i]
        } else if center {
            x - means[i]
        } else {
            x
        }
    };
    let stat = ["fit", "residual", "q", "bartlett", "regression", "score"].iter().find(|s| cmd.opts.iter().any(|(n, _)| n.eq_ignore_ascii_case(s) || (n.len() >= 3 && s.starts_with(&n.to_lowercase())))).copied().unwrap_or("");
    let mut out = Vec::new();
    let n = ds.n_obs;
    if which == "pca" && matches!(stat, "fit" | "residual" | "q") {
        let (ty, newn) = mvutil::new_var_names(ds, &cmd.main, if stat == "q" { 1 } else { p })?;
        // projection on the row space of L: P = L (L'L)^- L'
        let ltl = l.transpose().mul(&l).unwrap();
        let pr = l.mul(&mvutil::sym_inv(&ltl)).unwrap().mul(&l.transpose()).unwrap();
        let zc = |i: usize, t: usize| -> f64 { if corr { (cols[i][t] - means[i]) / sds[i] } else { cols[i][t] - means[i] } };
        let mut fits = vec![vec![0.0; rows.len()]; p];
        for (k, &t) in rows.iter().enumerate() {
            for i in 0..p {
                let s: f64 = (0..p).map(|j| zc(j, t) * pr[(j, i)]).sum();
                fits[i][k] = if corr { s * sds[i] + means[i] } else { s + means[i] };
            }
        }
        if stat == "q" {
            let q: Vec<f64> = (0..rows.len()).map(|k| (0..p).map(|i| { let r = (cols[i][rows[k]] - fits[i][k]) / if corr { sds[i] } else { 1.0 }; r * r }).sum()).collect();
            mvutil::put_var(ds, st, &newn[0], ty, n, &rows, &q, "")?;
        } else {
            if newn.len() != p {
                return Err(rc_err(198, format!("{p} new variables required")));
            }
            for i in 0..p {
                let vals: Vec<f64> = if stat == "fit" { fits[i].clone() } else { (0..rows.len()).map(|k| cols[i][rows[k]] - fits[i][k]).collect() };
                mvutil::put_var(ds, st, &newn[i], ty, n, &rows, &vals, &format!("{} {}", if stat == "fit" { "Fitted" } else { "Residual" }, names[i]))?;
            }
        }
        st.data_changed = true;
        return Ok(Some(String::new()));
    }
    let (ty, newn) = mvutil::new_var_names(ds, &cmd.main, f)?;
    let k = newn.len();
    // scoring coefficients (p x f)
    let (b, title, sub) = if which == "pca" {
        if stat.is_empty() {
            out.push("(score assumed)".into());
        }
        let ltl = l.transpose().mul(&l).unwrap();
        (l.mul(&mvutil::sym_inv(&ltl)).unwrap(), "Scoring coefficients ".to_string(), "    sum of squares(column-loading) = 1".to_string())
    } else {
        let c = mat(&e, "C")?;
        if stat == "bartlett" {
            let psi = evec(&e, "Psi");
            let mut pl = l.clone();
            for i in 0..p {
                for j in 0..f {
                    pl[(i, j)] /= psi[i];
                }
            }
            let gam = l.transpose().mul(&pl).unwrap();
            (pl.mul(&mvutil::sym_inv(&gam)).unwrap(), "Scoring coefficients (method = Bartlett)".to_string(), String::new())
        } else {
            if stat.is_empty() {
                out.push("(option regression assumed; regression scoring)".into());
            }
            (mvutil::sym_inv(&c).mul(&l).unwrap().mul(&phi).unwrap(), "Scoring coefficients (method = regression)".to_string(), String::new())
        }
    };
    if k < f {
        out.push(format!("({} {} skipped)", f - k, if which == "pca" { if f - k == 1 { "component" } else { "components" } } else if f - k == 1 { "factor" } else { "factors" }));
    }
    let pre = if which == "pca" { "Comp" } else { "Factor" };
    for j in 0..k {
        let vals: Vec<f64> = rows
            .iter()
            .map(|&t| if j < f { (0..p).map(|i| z(i, t) * b[(i, j)]).sum() } else { SYSMISS })
            .collect();
        let lab = if which == "pca" { format!("Scores for component {}", j + 1) } else { format!("Scores for factor {}", j + 1) };
        mvutil::put_var(ds, st, &newn[j], ty, n, &rows, &vals, &lab)?;
    }
    st.data_changed = true;
    if !cmd.has_opt("notable", 5) {
        out.push(String::new());
        out.push(title);
        if !sub.is_empty() {
            out.push(sub);
        }
        out.push(String::new());
        let fm: &'static str = if which == "pca" { "%8.4f" } else { "%8.5f" };
        let cell = move |v: f64| fmt(v, fm).trim().to_string();
        out.extend(loading_table(&names, &mvutil::names(pre, f), &b, &cell, None, 10));
    }
    let r = st.begin_r();
    r.set_mat("scoef", b.with_names(names.clone(), mvutil::names(pre, f)));
    if which == "factor" {
        r.set_str("method", if stat == "bartlett" { "Bartlett" } else { "regression" });
    }
    Ok(Some(out.join("\n")))
}

// ---------------------------------------------------------------------------
// estat
// ---------------------------------------------------------------------------

fn anti(c: &Matrix) -> (Matrix, Matrix) {
    let r = mvutil::cov_to_corr(c);
    let ri = mvutil::sym_inv(&r);
    let p = r.rows;
    let mut acorr = ri.clone();
    let mut acov = ri.clone();
    for i in 0..p {
        for j in 0..p {
            acorr[(i, j)] = ri[(i, j)] / (ri[(i, i)] * ri[(j, j)]).sqrt();
            acov[(i, j)] = ri[(i, j)] / (ri[(i, i)] * ri[(j, j)]);
        }
    }
    (acorr.with_names(c.rownames.clone(), c.colnames.clone()), acov.with_names(c.rownames.clone(), c.colnames.clone()))
}

fn one_col_table(title: &str, names: &[String], head: &str, vals: &[f64], fmt_: &str, extra: Option<(&str, f64)>) -> Vec<String> {
    let mut out = vec![String::new(), title.to_string(), String::new()];
    let mut b = Boxed::new("Variable", vec![vec![(head.to_string(), 10)]]);
    for (i, nm) in names.iter().enumerate() {
        b.row(nm, vec![fmt(vals[i], fmt_).trim().to_string()]);
    }
    if let Some((lab, v)) = extra {
        b.seps.push(names.len() - 1);
        b.row(lab, vec![fmt(v, fmt_).trim().to_string()]);
    }
    out.extend(b.render());
    out
}

pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    let which = is_mv(st)?;
    let r = (|| -> Result<Option<String>, String> {
        let e = st.e.clone();
        let names = vnames(&e);
        let p = names.len();
        let c = mat(&e, "C")?;
        let fmt_ = |d: &str| cmd.opt("format", 3).map(|s| s.trim().to_string()).unwrap_or_else(|| d.to_string());
        let mut out = Vec::new();
        match sub {
            "anti" => {
                let (acorr, acov) = anti(&c);
                if !cmd.has_opt("nocorr", 6) {
                    out.push(String::new());
                    out.push("Anti-image correlation coefficients    partialing out all other variables".into());
                    out.push(String::new());
                    out.extend(mvutil::matlist(&acorr, "Variable", &fmt_("%7.4f"), 8, true, 4));
                }
                if !cmd.has_opt("nocov", 5) {
                    out.push(String::new());
                    out.push("Anti-image covariance coefficients    partialing out all other variables".into());
                    out.push(String::new());
                    out.extend(mvutil::matlist(&acov, "Variable", &fmt_("%7.4f"), 8, true, 4));
                }
                let r = st.begin_r();
                r.set_mat("acorr", acorr);
                r.set_mat("acov", acov);
            }
            "kmo" => {
                let rr = mvutil::cov_to_corr(&c);
                let (a, _) = anti(&c);
                let mut kw = vec![0.0; p];
                let (mut sr, mut sa) = (0.0, 0.0);
                for i in 0..p {
                    let (mut r2, mut a2) = (0.0, 0.0);
                    for j in 0..p {
                        if i != j {
                            r2 += rr[(i, j)].powi(2);
                            a2 += a[(i, j)].powi(2);
                        }
                    }
                    kw[i] = r2 / (r2 + a2);
                    sr += r2;
                    sa += a2;
                }
                let kmo = sr / (sr + sa);
                if cmd.has_opt("novar", 5) {
                    out.push(String::new());
                    out.push(format!("Kaiser-Meyer-Olkin measure of sampling adequacy = {}", fmt(kmo, &fmt_("%8.4f")).trim()));
                } else {
                    out.extend(one_col_table("Kaiser-Meyer-Olkin measure of sampling adequacy", &names, "kmo", &kw, &fmt_("%8.4f"), Some(("Overall", kmo))));
                }
                let r = st.begin_r();
                r.set("kmo", kmo);
                r.set_mat("kmow", Matrix::col_vector(&kw).with_names(names.clone(), vec!["kmo".into()]));
            }
            "smc" => {
                let s = smc_of(&mvutil::cov_to_corr(&c));
                out.extend(one_col_table("Squared multiple correlations of variables with all other variables", &names, "smc", &s, &fmt_("%8.4f"), None));
                let r = st.begin_r();
                r.set_mat("smc", Matrix::col_vector(&s).with_names(names.clone(), vec!["smc".into()]));
            }
            "loadings" | "loading" | "load" => {
                if which != "pca" {
                    return Ok(None);
                }
                let (l, _, rot) = current_loadings(&e, cmd);
                let f = l.cols;
                let norm = cmd.opt("cnorm", 2).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "unit".into());
                let ev: Vec<f64> = if rot { evec(&e, "r_Ev") } else { evec(&e, "Ev") };
                let mut a = l.clone();
                for j in 0..f {
                    let s = match norm.as_str() {
                        "eigen" => ev[j].sqrt(),
                        "inveigen" => 1.0 / ev[j].sqrt(),
                        _ => 1.0,
                    };
                    for i in 0..p {
                        a[(i, j)] *= s;
                    }
                }
                out.push(String::new());
                out.push(format!("Principal component loadings ({})", if rot { "rotated" } else { "unrotated" }));
                out.push(format!(
                    "    component normalization: sum of squares(column) = {}",
                    match norm.as_str() {
                        "eigen" => "eigenvalue",
                        "inveigen" => "1/eigenvalue",
                        _ => "1",
                    }
                ));
                out.push(String::new());
                let fm = fmt_("%8.4g");
                let cell = move |v: f64| if fm.ends_with('g') { sig(v, 4) } else { fmt(v, &fm).trim().to_string() };
                out.extend(loading_table(&names, &mvutil::names("Comp", f), &a, &cell, None, 10));
                let r = st.begin_r();
                r.set_str("cnorm", norm.clone());
                r.set_mat("A", a.with_names(names.clone(), mvutil::names("Comp", f)));
            }
            "residuals" | "residual" | "res" | "resid" => {
                let (l, phi, _) = current_loadings(&e, cmd);
                let fit = if which == "pca" {
                    let ev = evec(&e, "Ev");
                    let lu = mat(&e, "L")?;
                    let f = lu.cols;
                    let mut fm = Matrix::zeros(p, p);
                    for i in 0..p {
                        for j in 0..p {
                            fm[(i, j)] = (0..f).map(|k| lu[(i, k)] * ev[k] * lu[(j, k)]).sum();
                        }
                    }
                    fm
                } else {
                    let psi = evec(&e, "Psi");
                    let mut fm = l.mul(&phi).unwrap().mul(&l.transpose()).unwrap();
                    for i in 0..p {
                        fm[(i, i)] += psi[i];
                    }
                    fm
                }
                .with_names(names.clone(), names.clone());
                let res = c.sub(&fit).unwrap().with_names(names.clone(), names.clone());
                let df = fmt_("%7.4f");
                let kind = if which == "pca" { e.macros.get("Ctype").cloned().unwrap_or_default() } else { "correlation".into() };
                if cmd.has_opt("obs", 3) {
                    out.push(String::new());
                    out.push(if which == "pca" { format!("Observed {kind} matrix") } else { "Observed correlations".into() });
                    out.push(String::new());
                    out.extend(mvutil::matlist(&c, "Variable", &df, 8, true, 4));
                }
                if cmd.has_opt("fitted", 3) {
                    out.push(String::new());
                    out.push(if which == "pca" { format!("Fitted {kind} matrix") } else { "Fitted (\"reconstructed\") values for correlations".into() });
                    out.push(String::new());
                    out.extend(mvutil::matlist(&fit, "Variable", &df, 8, true, 4));
                }
                let r = st.begin_r();
                r.set_mat("fit", fit.clone());
                if which == "factor" && cmd.has_opt("sresiduals", 4) {
                    let n = e.get("N").unwrap_or(0.0);
                    let mut sr = res.clone();
                    for i in 0..p {
                        for j in 0..p {
                            sr[(i, j)] = n.sqrt() * res[(i, j)] / (fit[(i, j)].powi(2) + fit[(i, i)] * fit[(j, j)]).sqrt();
                        }
                    }
                    out.push(String::new());
                    out.push("Standardized residuals of correlations".into());
                    out.push(String::new());
                    out.extend(mvutil::matlist(&sr, "Variable", &df, 8, true, 4));
                    r.set_mat("SR", sr);
                } else {
                    out.push(String::new());
                    out.push(if which == "pca" { format!("Residual {kind} matrix") } else { "Raw residuals of correlations (observed-fitted)".into() });
                    out.push(String::new());
                    out.extend(mvutil::matlist(&res, "Variable", &df, 8, true, 4));
                }
                r.set_mat(if which == "pca" { "residual" } else { "res" }, res);
            }
            "rotatecompare" | "rotatec" | "rotatecomp" => {
                if !has_rot(&e) {
                    return Err(rc_err(321, "no rotation results available"));
                }
                let t = mat(&e, "r_T")?;
                let pre = if which == "pca" { "Comp" } else { "Factor" };
                out.push(String::new());
                out.push(format!("Rotation matrix -- {}", mv_rotate::rotation_text(&e)));
                out.push(String::new());
                out.extend(rot_matrix_table(&t, pre));
                let lr = mat(&e, "r_L")?;
                let lu = mat(&e, "L")?;
                let f = lu.cols;
                let cell = |v: f64| format!("{v:.4}");
                if which == "pca" {
                    out.push(String::new());
                    out.push("Rotated component loadings".into());
                    out.push(String::new());
                    out.extend(loading_table(&names, &mvutil::names(pre, f), &lr, &cell, None, 10));
                    out.push(String::new());
                    out.push("Unrotated component loadings".into());
                    out.push(String::new());
                    out.extend(loading_table(&names, &mvutil::names(pre, f), &lu, &cell, None, 10));
                } else {
                    out.push(String::new());
                    out.push("Factor loadings".into());
                    out.push(String::new());
                    let cn = mvutil::names(pre, f);
                    let mut b = Boxed::new("Variable", vec![cn.iter().map(|c| (c.clone(), 10)).collect(), cn.iter().map(|c| (c.clone(), 10)).collect()]);
                    for i in 0..p {
                        let mut cells: Vec<String> = (0..f).map(|j| cell(lr[(i, j)])).collect();
                        cells.extend((0..f).map(|j| cell(lu[(i, j)])));
                        b.row(&names[i], cells);
                    }
                    let mut lines = b.render();
                    let w = 10 * f;
                    lines.insert(1, format!("    {:12}   {:^w$}  {:^w$}", "", "Rotated", "Unrotated"));
                    out.extend(lines);
                }
            }
            "common" => {
                if which != "factor" {
                    return Ok(None);
                }
                let (_, phi, rot) = current_loadings(&e, cmd);
                let k = phi.cols;
                out.push(String::new());
                out.push(if rot && e.macros.get("r_class").map(String::as_str) == Some("oblique") {
                    format!("Correlation matrix of the {} rotated common factors", e.macros.get("r_ctitle").cloned().unwrap_or_default())
                } else {
                    "Correlation matrix of the common factors".into()
                });
                out.push(String::new());
                let cn = mvutil::names("Factor", k);
                let mut b = Boxed::new("Factors", vec![cn.iter().map(|c| (c.clone(), 10)).collect()]);
                for i in 0..k {
                    b.row(&cn[i], (0..=i).map(|j| sig(phi[(i, j)], 4)).collect());
                }
                out.extend(b.render());
                st.begin_r().set_mat("Phi", phi.with_names(cn.clone(), cn));
            }
            "structure" => {
                if which != "factor" {
                    return Ok(None);
                }
                let (l, phi, _) = current_loadings(&e, cmd);
                let s = l.mul(&phi).unwrap();
                let f = s.cols;
                out.push(String::new());
                out.push("Structure matrix: correlations between variables and common factors".into());
                out.push(String::new());
                let cell = |v: f64| format!("{v:.4}");
                out.extend(loading_table(&names, &mvutil::names("Factor", f), &s, &cell, None, 10));
                st.begin_r().set_mat("st", s.with_names(names.clone(), mvutil::names("Factor", f)));
            }
            "factors" => {
                if which != "factor" {
                    return Ok(None);
                }
                let n = e.get("N").unwrap_or(0.0);
                let rr = mvutil::cov_to_corr(&c);
                let maxq = (1..=p).take_while(|&q| p * q - q * (q - 1) / 2 <= p * (p - 1) / 2).last().unwrap_or(1);
                let kmax = mvutil::opt_usize(cmd, "factors", 3)?.unwrap_or(maxq).min(maxq);
                let mut rows = Vec::new();
                let mut hey = false;
                for q in 1..=kmax {
                    let (psi, _, fmin, _) = ml_fit(&rr, q, None);
                    if psi.iter().any(|&x| x <= 0.0051) {
                        hey = true;
                    }
                    let ll = -n / 2.0 * fmin;
                    let dfm = (p * q - q * (q - 1) / 2) as f64;
                    let dfr = (p * (p - 1) / 2) as f64 - dfm;
                    rows.push(vec![ll, dfm, dfr, -2.0 * ll + 2.0 * dfm, -2.0 * ll + dfm * n.ln()]);
                }
                out.push(String::new());
                out.push("Factor analysis with different numbers of factors (maximum likelihood)".into());
                out.push(String::new());
                out.push(format!("    {}", "-".repeat(63)));
                out.push(format!("    {:>9} | {:>12} {:>8} {:>8} {:>12} {:>12}", "#factors", "loglik", "df_m", "df_r", "AIC", "BIC"));
                out.push(format!("    {}+{}", "-".repeat(10), "-".repeat(52)));
                for (i, r) in rows.iter().enumerate() {
                    out.push(format!("    {:>9} | {:>12} {:>8} {:>8} {:>12} {:>12}", i + 1, sig(r[0], 7), r[1], r[2], sig(r[3], 7), sig(r[4], 7)));
                }
                out.push(format!("    {}", "-".repeat(63)));
                out.push(format!("    {}", if hey { "Heywood cases encountered" } else { "no Heywood cases encountered" }));
                let m = Matrix::from_rows(&rows).with_names((1..=rows.len()).map(|i| i.to_string()).collect(), vec!["loglik".into(), "df_m".into(), "df_r".into(), "AIC".into(), "BIC".into()]);
                st.begin_r().set_mat("stats", m);
            }
            "summarize" | "summ" | "su" | "sum" => {
                let sample = e.sample.clone().ok_or_else(|| rc_err(321, "estat summarize not available after pcamat/factormat"))?;
                out.push(String::new());
                out.push("  Estimation sample pca".replace("pca", &which) + &format!("{:>38}", format!("Number of obs = {:>6}", util::commas(e.get("N").unwrap_or(0.0) as usize))));
                out.push(String::new());
                out.push(format!("  {}", "-".repeat(60)));
                out.push(format!("  {:>12} | {:>10} {:>10} {:>10} {:>10}", "Variable", "Mean", "Std. dev.", "Min", "Max"));
                out.push(format!("  {}+{}", "-".repeat(13), "-".repeat(46)));
                let mut stats = Vec::new();
                for v in &names {
                    let col = util::col(ds, v)?;
                    let xs: Vec<f64> = (0..ds.n_obs).filter(|&i| sample.get(i).copied().unwrap_or(false)).map(|i| col[i]).collect();
                    let m = util::mean(&xs);
                    let sd = util::variance(&xs).sqrt();
                    let (mn, mx) = xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, &x| (a.0.min(x), a.1.max(x)));
                    out.push(format!("  {:>12} | {:>10} {:>10} {:>10} {:>10}", crate::summarize::abbrev(v, 12), mvutil::g(m, 9), mvutil::g(sd, 9), mvutil::g(mn, 9), mvutil::g(mx, 9)));
                    stats.push(vec![m, sd, mn, mx]);
                }
                out.push(format!("  {}", "-".repeat(60)));
                st.begin_r().set_mat("stats", Matrix::from_rows(&stats).with_names(names.clone(), vec!["mean".into(), "sd".into(), "min".into(), "max".into()]));
            }
            "vce" => return Ok(None),
            _ => return Ok(None),
        }
        Ok(Some(out.join("\n")))
    })();
    match r {
        Ok(Some(s)) => Some(Ok(s)),
        Ok(None) => {
            if st.e.matrices.contains_key("V") {
                None
            } else {
                Some(Err(rc_err(321, format!("estat {sub} not valid after {which}"))))
            }
        }
        Err(e) => Some(Err(e)),
    }
}

// ---------------------------------------------------------------------------
// plots
// ---------------------------------------------------------------------------

pub struct Screeplot;

impl CommandHandler for Screeplot {
    fn name(&self) -> &'static str {
        "screeplot"
    }

    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        // after pca/factor (or any e(Ev)), or of a given matrix
        let ev: Vec<f64> = if !cmd.main.trim().is_empty() {
            st.matrices.get(cmd.main.trim()).map(|m| m.data.clone()).ok_or_else(|| rc_err(111, format!("matrix {} not found", cmd.main.trim())))?
        } else if let Some(m) = st.e.matrices.get("Ev") {
            m.data.clone()
        } else if let Some(m) = st.e.matrices.get("Eigen") {
            m.data.clone()
        } else {
            return Err(rc_err(301, "last estimates not found"));
        };
        let k = mvutil::opt_usize(cmd, "neigen", 2)?.unwrap_or(ev.len()).min(ev.len());
        let mut g = Graph { title: "Scree plot of eigenvalues after ".to_string() + st.e.macros.get("cmd").map(String::as_str).unwrap_or(""), xtitle: "Number".into(), ytitle: "Eigenvalues".into(), ..Default::default() };
        let pts: Vec<(f64, f64)> = (0..k).map(|i| ((i + 1) as f64, ev[i])).collect();
        g.layers.push(Layer { kind: LayerKind::Connected, points: pts, label: "Eigenvalues".into(), ..Default::default() });
        if cmd.has_opt("mean", 1) {
            g.ylines.push(ev.iter().sum::<f64>() / ev.len() as f64);
        }
        if cmd.has_opt("ci", 2) {
            let n = st.e.get("N").unwrap_or(0.0);
            if n > 0.0 {
                let z = dist::invnormal(0.975);
                let ranges: Vec<(f64, f64, f64)> = (0..k).map(|i| { let s = ev[i] * (2.0 / n).sqrt(); ((i + 1) as f64, ev[i] - z * s, ev[i] + z * s) }).collect();
                g.layers.insert(0, Layer { kind: LayerKind::Range(crate::graph::RangeStyle::Area), ranges, label: "95% CI".into(), alpha: 0.3, ..Default::default() });
            }
        }
        g.command = cmd.raw.clone();
        finish(&mut g, cmd, st)
    }
}

fn pair_graph(points: Vec<(f64, f64)>, labels: Vec<String>, xt: &str, yt: &str, title: &str) -> Graph {
    let mut g = Graph { title: title.into(), xtitle: xt.into(), ytitle: yt.into(), ..Default::default() };
    g.layers.push(Layer { kind: LayerKind::Scatter, points, mlabels: labels, ..Default::default() });
    g
}

pub struct Loadingplot {
    pub score: bool,
}

impl CommandHandler for Loadingplot {
    fn name(&self) -> &'static str {
        if self.score { "scoreplot" } else { "loadingplot" }
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let which = is_mv(st).ok_or_else(|| rc_err(321, "last estimates not found"))?;
        let e = st.e.clone();
        let (l, _, rot) = current_loadings(&e, cmd);
        let names = vnames(&e);
        let k = mvutil::opt_usize(cmd, "factors", 3)?.or(mvutil::opt_usize(cmd, "components", 4)?).unwrap_or(2).min(l.cols).max(1);
        let pre = if which == "pca" { "Component" } else { "Factor" };
        // loadings: pca shows eigenvectors scaled per cnorm (unit by default)
        let pairs: Vec<(usize, usize)> = if k < 2 { vec![(0, 0)] } else { (0..k).flat_map(|a| (a + 1..k).map(move |b| (a, b))).collect() };
        let mut graphs = Vec::new();
        let title = if self.score {
            format!("Score variables ({}{})", which, if rot { ", rotated" } else { "" })
        } else if which == "pca" {
            format!("Component loadings{}", if rot { " (rotated)" } else { "" })
        } else {
            format!("Factor loadings{}", if rot { " (rotated)" } else { "" })
        };
        let scores: Option<Vec<Vec<f64>>> = if self.score {
            let mut tmp = ds.clone();
            let mut st2 = st.clone();
            let stub = "__mvscore";
            let mut c2 = cmd.clone();
            c2.main = (1..=k).map(|j| format!("{stub}{j}")).collect::<Vec<_>>().join(" ");
            c2.opts.retain(|(n, _)| n.eq_ignore_ascii_case("norotated"));
            c2.opts.push(("notable".into(), None));
            predict(&mut tmp, &mut st2, &c2)?;
            Some((1..=k).map(|j| util::col(&tmp, &format!("{stub}{j}"))).collect::<Result<_, _>>()?)
        } else {
            None
        };
        for (a, b) in pairs {
            let (pts, labs): (Vec<(f64, f64)>, Vec<String>) = match &scores {
                Some(s) => (0..s[a].len()).filter(|&i| !is_missing(s[a][i]) && !is_missing(s[b][i])).map(|i| ((s[a][i], s[b][i]), String::new())).unzip(),
                None => (0..names.len()).map(|i| ((l[(i, a)], l[(i, b)]), names[i].clone())).unzip(),
            };
            graphs.push(pair_graph(pts, labs, &format!("{} {}", if self.score { "Scores for" } else { "" }, format!("{pre} {}", a + 1)).trim().to_string(), &format!("{} {}", if self.score { "Scores for" } else { "" }, format!("{pre} {}", b + 1)).trim().to_string(), ""));
        }
        let mut g = if graphs.len() == 1 {
            let mut g = graphs.pop().unwrap();
            g.title = title;
            g
        } else {
            Graph { title, children: graphs, cols: 2, ..Default::default() }
        };
        g.command = cmd.raw.clone();
        let _ = SYSMISS;
        finish(&mut g, cmd, st)
    }
}
