//! Multidimensional scaling (`mds`, `mdsmat`, `mdslong`), correspondence
//! analysis (`ca`, `camat`, `mca`) and `biplot`.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::graph::{finish, Graph, Layer, LayerKind, PcStyle};
use crate::matrix::Matrix;
use crate::mv_cluster::Measure;
use crate::mvutil::{self, sig};
use crate::store::Store;
use crate::util;

// ---------------------------------------------------------------------------
// MDS core
// ---------------------------------------------------------------------------

struct MdsOpts {
    method: String,
    loss: String,
    transform: String,
    dim: usize,
    neigen: usize,
    addconstant: bool,
    maxit: usize,
}

fn mds_opts(cmd: &Command) -> Result<MdsOpts, String> {
    let mut method = cmd.opt("method", 4).map(|s| s.trim().to_lowercase()).unwrap_or_default();
    let mut loss = cmd.opt("loss", 4).map(|s| s.trim().to_lowercase()).unwrap_or_default();
    let mut transform = cmd.opt("transform", 5).map(|s| s.trim().to_lowercase()).unwrap_or_default();
    if transform.starts_with("mono") {
        transform = "monotonic".into();
    } else if transform.starts_with("ident") {
        transform = "identity".into();
    } else if transform.starts_with("pow") {
        transform = "power".into();
    }
    if method.starts_with("non") {
        method = "nonmetric".into();
        loss = "stress".into();
        transform = "monotonic".into();
    } else if method.starts_with("mod") {
        method = "modern".into();
    } else if method.starts_with("cla") || (method.is_empty() && loss.is_empty() && transform.is_empty()) {
        method = "classical".into();
    } else {
        method = if transform == "monotonic" && (loss.is_empty() || loss == "stress") { "nonmetric".into() } else { "modern".into() };
    }
    if method != "classical" {
        if loss.is_empty() {
            loss = "stress".into();
        }
        if transform.is_empty() {
            transform = "identity".into();
        }
    }
    Ok(MdsOpts {
        method,
        loss,
        transform,
        dim: mvutil::opt_usize(cmd, "dimension", 3)?.unwrap_or(2).max(1),
        neigen: mvutil::opt_usize(cmd, "neigen", 4)?.unwrap_or(10),
        addconstant: cmd.has_opt("addconstant", 4),
        maxit: mvutil::opt_usize(cmd, "iterate", 4)?.unwrap_or(5000),
    })
}

fn dist_of(y: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = y.len();
    let mut b = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..i {
            let d = y[i].iter().zip(&y[j]).map(|(a, c)| (a - c) * (a - c)).sum::<f64>().sqrt();
            b[i][j] = d;
            b[j][i] = d;
        }
    }
    b
}

/// Classical scaling: (eigenvalues, configuration n x p, addcons)
fn classical(d: &[Vec<f64>], p: usize, addconstant: bool) -> (Vec<f64>, Vec<Vec<f64>>, f64) {
    let n = d.len();
    let mut a = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = -0.5 * d[i][j] * d[i][j];
        }
    }
    let center = |a: &Matrix| -> Matrix {
        let rm: Vec<f64> = (0..n).map(|i| (0..n).map(|j| a[(i, j)]).sum::<f64>() / n as f64).collect();
        let gm = rm.iter().sum::<f64>() / n as f64;
        let mut b = a.clone();
        for i in 0..n {
            for j in 0..n {
                b[(i, j)] = a[(i, j)] - rm[i] - rm[j] + gm;
            }
        }
        b
    };
    let mut b = center(&a);
    let (mut vals, mut vecs) = b.sym_eigen();
    let mut addc = 0.0;
    if addconstant && vals[n - 1] < -1e-10 {
        addc = -2.0 * vals[n - 1];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    a[(i, j)] -= 0.5 * addc;
                }
            }
        }
        b = center(&a);
        let e = b.sym_eigen();
        vals = e.0;
        vecs = e.1;
    }
    let y: Vec<Vec<f64>> = (0..n).map(|i| (0..p).map(|k| vecs[(i, k)] * vals[k].max(0.0).sqrt()).collect()).collect();
    (vals, y, addc)
}

/// Center, rotate to principal axes, sign so that the first object (lowest id) is positive.
fn principal_norm(y: &mut [Vec<f64>], first: usize) {
    let n = y.len();
    let p = y.first().map(|r| r.len()).unwrap_or(0);
    for k in 0..p {
        let m = y.iter().map(|r| r[k]).sum::<f64>() / n as f64;
        for r in y.iter_mut() {
            r[k] -= m;
        }
    }
    let mut c = Matrix::zeros(p, p);
    for r in y.iter() {
        for a in 0..p {
            for b in 0..p {
                c[(a, b)] += r[a] * r[b];
            }
        }
    }
    let (_, v) = c.sym_eigen();
    for r in y.iter_mut() {
        let old = r.clone();
        for k in 0..p {
            r[k] = (0..p).map(|a| old[a] * v[(a, k)]).sum();
        }
    }
    for k in 0..p {
        if y[first][k] < 0.0 {
            for r in y.iter_mut() {
                r[k] = -r[k];
            }
        }
    }
}

/// Pool-adjacent-violators: monotone (nondecreasing in `order`) fit of `x`.
fn isotonic(x: &[f64]) -> Vec<f64> {
    let mut vals: Vec<f64> = Vec::new();
    let mut wts: Vec<f64> = Vec::new();
    for &v in x {
        vals.push(v);
        wts.push(1.0);
        while vals.len() > 1 && vals[vals.len() - 2] > vals[vals.len() - 1] {
            let (v2, w2) = (vals.pop().unwrap(), wts.pop().unwrap());
            let (v1, w1) = (vals.pop().unwrap(), wts.pop().unwrap());
            vals.push((v1 * w1 + v2 * w2) / (w1 + w2));
            wts.push(w1 + w2);
        }
    }
    let mut out = Vec::with_capacity(x.len());
    for (v, w) in vals.iter().zip(&wts) {
        for _ in 0..(*w as usize) {
            out.push(*v);
        }
    }
    out
}

fn loss_value(loss: &str, dh: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let n = b.len();
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..n {
        for j in 0..i {
            let (bb, d) = (b[i][j], dh[i][j]);
            match loss {
                "nstress" => {
                    num += (bb - d).powi(2);
                    den += d * d;
                }
                "sstress" => {
                    num += (bb * bb - d * d).powi(2);
                    den += bb.powi(4);
                }
                "nsstress" => {
                    num += (bb * bb - d * d).powi(2);
                    den += d.powi(4);
                }
                "sammon" => {
                    if d > 0.0 {
                        num += (bb - d).powi(2) / d;
                    }
                    den += d;
                }
                _ => {
                    num += (bb - d).powi(2);
                    den += bb * bb;
                }
            }
        }
    }
    if loss == "sammon" { num / den } else { (num / den).sqrt() }
}

/// Modern / nonmetric scaling: returns (configuration, loss, iterations, disparities, alpha).
fn modern(d: &[Vec<f64>], o: &MdsOpts, start: Vec<Vec<f64>>, log: &mut Vec<String>) -> (Vec<Vec<f64>>, f64, usize, Vec<Vec<f64>>, f64) {
    let n = d.len();
    let p = o.dim;
    let mut y = start;
    let mut dh: Vec<Vec<f64>> = d.to_vec();
    let mut alpha = 1.0;
    // order of the dissimilarities for isotonic regression
    let mut pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (0..i).map(move |j| (i, j))).collect();
    pairs.sort_by(|a, b| d[a.0][a.1].total_cmp(&d[b.0][b.1]));
    let mut last = f64::INFINITY;
    let mut it = 0;
    if o.loss == "strain" && o.transform == "identity" {
        let b = dist_of(&y);
        let l = strain_of(d, &y);
        let _ = b;
        log.push(format!("Iteration 1:   strain = {}", sig(l, 8)));
        log.push(format!("Iteration 2:   strain = {}", sig(l, 8)));
        return (y, l, 2, dh, alpha);
    }
    while it < o.maxit {
        it += 1;
        // C-step
        let b = dist_of(&y);
        let squared = o.loss.contains("sstress");
        if o.loss == "sammon" || squared {
            // gradient descent with backtracking
            let f0 = loss_value(&o.loss, &dh, &b);
            let mut g = vec![vec![0.0; p]; n];
            for i in 0..n {
                for j in 0..n {
                    if i == j || b[i][j] <= 1e-12 {
                        continue;
                    }
                    let coef = if squared { 4.0 * (b[i][j] * b[i][j] - dh[i][j] * dh[i][j]) } else if dh[i][j] > 0.0 { 2.0 * (b[i][j] - dh[i][j]) / (dh[i][j] * b[i][j]) } else { 0.0 };
                    for k in 0..p {
                        g[i][k] += coef * (y[i][k] - y[j][k]);
                    }
                }
            }
            let gn: f64 = g.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
            let scale: f64 = y.iter().flatten().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
            let mut step = 0.1 * scale / gn.max(1e-300);
            let mut moved = false;
            for _ in 0..40 {
                let yc: Vec<Vec<f64>> = (0..n).map(|i| (0..p).map(|k| y[i][k] - step * g[i][k]).collect()).collect();
                if loss_value(&o.loss, &dh, &dist_of(&yc)) < f0 {
                    y = yc;
                    moved = true;
                    break;
                }
                step /= 2.0;
            }
            if !moved {
                break;
            }
        } else {
            // Guttman transform (SMACOF)
            // ny = (1/n) * (B(Y) Y) for centered Y; recompute exactly
            let mut gy = vec![vec![0.0; p]; n];
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let r = if b[i][j] > 1e-12 { dh[i][j] / b[i][j] } else { 0.0 };
                    for k in 0..p {
                        gy[i][k] += r * (y[i][k] - y[j][k]);
                    }
                }
                for k in 0..p {
                    gy[i][k] /= n as f64;
                }
            }
            y = gy;
        }
        // T-step
        let b = dist_of(&y);
        if o.transform == "monotonic" {
            let xs: Vec<f64> = pairs.iter().map(|&(i, j)| b[i][j]).collect();
            let fit = isotonic(&xs);
            let ss: f64 = fit.iter().map(|v| v * v).sum();
            let sb: f64 = xs.iter().map(|v| v * v).sum();
            let sc = if ss > 0.0 { (sb / ss).sqrt() } else { 1.0 };
            for (t, &(i, j)) in pairs.iter().enumerate() {
                dh[i][j] = fit[t] * sc;
                dh[j][i] = fit[t] * sc;
            }
        } else if o.transform == "power" {
            // golden section on alpha in (0.01, 5]
            let f = |a: f64| -> f64 {
                let dd: Vec<Vec<f64>> = d.iter().map(|r| r.iter().map(|v| v.powf(a)).collect()).collect();
                loss_value(&o.loss, &dd, &b)
            };
            let (mut lo, mut hi) = (0.01, 5.0);
            let gr = 0.618_033_988_75;
            for _ in 0..60 {
                let a1 = hi - gr * (hi - lo);
                let a2 = lo + gr * (hi - lo);
                if f(a1) < f(a2) {
                    hi = a2;
                } else {
                    lo = a1;
                }
            }
            alpha = 0.5 * (lo + hi);
            dh = d.iter().map(|r| r.iter().map(|v| v.powf(alpha)).collect()).collect();
        }
        let l = loss_value(&o.loss, &dh, &b);
        log.push(format!("Iteration {it}:   {} = {}", o.loss, sig(l, 8)));
        if (last - l).abs() < 1e-12 * l.max(1e-12) || (last - l).abs() < 1e-15 {
            last = l;
            break;
        }
        last = l;
    }
    (y, last, it, dh, alpha)
}

fn strain_of(d: &[Vec<f64>], y: &[Vec<f64>]) -> f64 {
    let n = d.len();
    let rm: Vec<f64> = (0..n).map(|i| (0..n).map(|j| -0.5 * d[i][j] * d[i][j]).sum::<f64>() / n as f64).collect();
    let gm = rm.iter().sum::<f64>() / n as f64;
    let mut s = 0.0;
    for i in 0..n {
        for j in 0..n {
            let b = -0.5 * d[i][j] * d[i][j] - rm[i] - rm[j] + gm;
            let yy: f64 = y[i].iter().zip(&y[j]).map(|(a, c)| a * c).sum();
            s += (b - yy).powi(2);
        }
    }
    s.sqrt()
}

#[allow(clippy::too_many_arguments)]
fn run_mds(st: &mut Store, d: Vec<Vec<f64>>, ids: Vec<String>, o: &MdsOpts, dtitle: &str, dname: &str, extra: &dyn Fn(&mut crate::store::Results), sample: Option<Vec<bool>>, noplot: bool, config: bool, cmd: &Command) -> Result<String, String> {
    let n = d.len();
    if n < 2 {
        return Err(rc_err(2001, "insufficient observations"));
    }
    let first = (0..n).min_by(|&a, &b| ids[a].cmp(&ids[b])).unwrap_or(0);
    let mut out = Vec::new();
    let (vals, mut y, addc) = classical(&d, o.dim.min(n), o.addconstant);
    let np = vals.iter().filter(|&&v| v > 1e-10 * vals[0].abs().max(1.0)).count();
    if o.method == "classical" && np < o.dim {
        return Err(rc_err(198, format!("dimension({}) exceeds the number of positive eigenvalues ({np})", o.dim)));
    }
    let p = o.dim.min(n);
    let mut crit = SYSMISS;
    let mut ic = 0;
    let mut disp = None;
    let mut alpha = SYSMISS;
    if o.method == "classical" {
        principal_norm(&mut y, first);
        let sa: f64 = vals.iter().map(|v| v.abs()).sum();
        let s2: f64 = vals.iter().map(|v| v * v).sum();
        let m1 = vals[..p].iter().map(|v| v.abs()).sum::<f64>() / sa;
        let m2 = vals[..p].iter().map(|v| v * v).sum::<f64>() / s2;
        out.push(String::new());
        out.push("Classical metric multidimensional scaling".into());
        out.push(format!("    {dtitle}"));
        out.push(String::new());
        out.push(format!("{:<49}Number of obs        = {:>6}", "", n));
        out.push(format!("    {:<24}= {:>8}{:13}Mardia fit measure 1 = {:>6.4}", "Eigenvalues > 0", np, "", m1));
        out.push(format!("    {:<24}= {:>8}{:13}Mardia fit measure 2 = {:>6.4}", "Retained dimensions", p, "", m2));
        out.push(String::new());
        out.push(format!("    {}", "-".repeat(74)));
        out.push(format!("    {:>12} |{:>16}{:>28}{:>22}", "", "", "abs(eigenvalue)", "(eigenvalue)^2"));
        out.push(format!("    {:>12} | {:>14}{:>18}{:>10}{:>12}{:>10}", "Dimension", "Eigenvalue", "Percent", "Cumul.", "Percent", "Cumul."));
        out.push(format!("    {}+{}", "-".repeat(13), "-".repeat(60)));
        let (mut c1, mut c2) = (0.0, 0.0);
        for k in 0..o.neigen.min(n).min(vals.len()) {
            if vals[k].abs() < 1e-12 * vals[0].abs() {
                break;
            }
            let p1 = 100.0 * vals[k].abs() / sa;
            let p2 = 100.0 * vals[k] * vals[k] / s2;
            c1 += p1;
            c2 += p2;
            out.push(format!("    {:>12} | {:>14}{:>18.2}{:>10.2}{:>12.2}{:>10.2}", k + 1, sig(vals[k], 8), p1, c1.min(100.0), p2, c2.min(100.0)));
        }
        out.push(format!("    {}", "-".repeat(74)));
        let e = st.begin_e();
        e.set("mardia1", m1);
        e.set("mardia2", m2);
    } else {
        let mut log = Vec::new();
        if o.method == "modern" && cmd.opt("transform", 5).is_none() {
            out.push("(transform(identity) assumed)".into());
        }
        let (yy, l, it, dh, a) = modern(&d, o, y.clone(), &mut log);
        y = yy;
        principal_norm(&mut y, first);
        crit = l;
        ic = it;
        alpha = a;
        if o.transform == "monotonic" {
            disp = Some(dh);
        }
        if !cmd.has_opt("nolog", 5) && !cmd.has_opt("protect", 4) {
            out.extend(log);
        }
        let losstitle = match o.loss.as_str() {
            "stress" => "stress = raw_stress/norm(distances)",
            "nstress" => "nstress = raw_stress/norm(disparities)",
            "sstress" => "sstress = raw_sstress/norm(distances^2)",
            "nsstress" => "nsstress = raw_sstress/norm(disparities^2)",
            "strain" => "strain = loss for classical MDS",
            _ => "sammon = sammon mapping",
        };
        let tt = match o.transform.as_str() {
            "identity" => "identity (no transformation)".to_string(),
            "power" => format!("power (alpha = {})", sig(alpha, 6)),
            _ => "monotonic (nonmetric)".to_string(),
        };
        out.push(String::new());
        out.push(if o.method == "nonmetric" { "Non-metric multidimensional scaling".into() } else { "Modern multidimensional scaling".into() });
        out.push(format!("    {dtitle}"));
        out.push(format!("      Loss criterion: {losstitle}"));
        out.push(format!("      Transformation: {tt}"));
        out.push(format!("{:<49}Number of obs        = {:>6}", "", n));
        out.push(format!("{:<49}Dimensions           = {:>6}", "", p));
        out.push(format!("{:<49}Loss criterion       = {:>9.4}", "      Normalization: principal", crit));
    }
    let names = mvutil::names("dim", p);
    let mut ym = Matrix::zeros(n, p);
    for i in 0..n {
        for k in 0..p {
            ym[(i, k)] = y[i][k];
        }
    }
    let ym = ym.with_names(ids.clone(), names.clone());
    let mut dm = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            dm[(i, j)] = d[i][j];
        }
    }
    if config {
        out.push(String::new());
        out.push("Configuration in 2-dimensional Euclidean space (principal normalization)".replace("2", &p.to_string()));
        out.push(String::new());
        out.extend(mvutil::matlist(&ym, "", "%9.4f", 9, false, 4));
    }
    let e = &mut st.e;
    if o.method != "classical" {
        e.set("critval", crit);
        e.set("ic", ic as f64);
        e.set("converged", 1.0);
        e.set("rc", 0.0);
        if o.transform == "power" {
            e.set("alpha", alpha);
        }
    }
    e.set("N", n as f64);
    e.set("p", p as f64);
    e.set("np", np as f64);
    e.set("addcons", addc);
    e.set_str("cmd", "mds");
    e.set_str("method", if o.method == "nonmetric" { "modern".to_string() } else { o.method.clone() });
    if o.method == "nonmetric" {
        e.set_str("method2", "nonmetric");
    }
    if o.method != "classical" {
        e.set_str("loss", o.loss.clone());
        e.set_str("tfunction", o.transform.clone());
    }
    e.set_str("norm", "principal");
    e.set_str("dname", dname);
    e.set_str("dtype", "dissimilarity");
    e.set_str("properties", if o.method == "classical" { "nob noV eigen" } else { "nob noV" });
    e.set_str("estat_cmd", "mds_estat");
    e.set_str("predict", "mds_p");
    e.set_str("marginsnotok", "_ALL");
    e.set_mat("Y", ym.clone());
    e.set_mat("D", dm.with_names(ids.clone(), ids.clone()));
    e.set_mat("Ev", Matrix::row_vector(&vals).with_names(vec!["Ev".into()], mvutil::names("dim", vals.len())));
    if let Some(dh) = disp {
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] = dh[i][j];
            }
        }
        e.set_mat("Disparities", m.with_names(ids.clone(), ids.clone()));
    }
    e.sample = sample;
    extra(e);
    if !noplot && p >= 2 {
        let _ = mdsconfig_graph(st, cmd);
    }
    Ok(out.join("\n"))
}

fn mdsconfig_graph(st: &mut Store, cmd: &Command) -> Result<String, String> {
    let y = st.e.matrices.get("Y").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let pts: Vec<(f64, f64)> = (0..y.rows).map(|i| (y[(i, 0)], y[(i, 1.min(y.cols - 1))])).collect();
    let mut g = Graph { title: "MDS configuration".into(), xtitle: "Dimension 1".into(), ytitle: "Dimension 2".into(), ..Default::default() };
    g.layers.push(Layer { kind: LayerKind::Scatter, points: pts, mlabels: y.rownames.clone(), ..Default::default() });
    g.command = cmd.raw.clone();
    let mut c = cmd.clone();
    c.opts.retain(|(n, _)| ["title", "xtitle", "ytitle", "name", "saving"].iter().any(|k| n.eq_ignore_ascii_case(k)));
    finish(&mut g, &c, st)
}

pub struct Mdsconfig;
impl CommandHandler for Mdsconfig {
    fn name(&self) -> &'static str {
        "mdsconfig"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        mdsconfig_graph(st, cmd)
    }
}

/// Dissimilarities from proximities: similarity conversion.
fn s2d(m: &mut [Vec<f64>], how: &str) {
    let n = m.len();
    let diag: Vec<f64> = (0..n).map(|i| m[i][i]).collect();
    for i in 0..n {
        for j in 0..n {
            m[i][j] = if how.starts_with('o') { 1.0 - m[i][j] } else { (diag[i] + diag[j] - 2.0 * m[i][j]).max(0.0).sqrt() };
        }
        m[i][i] = 0.0;
    }
}

pub struct Mds;

impl CommandHandler for Mds {
    fn name(&self) -> &'static str {
        "mds"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let idv = cmd.opt("id", 2).map(|s| s.trim().to_string()).ok_or_else(|| rc_err(198, "option id() required"))?;
        let vars = crate::varlist::expand(ds, &cmd.main)?;
        let o = mds_opts(cmd)?;
        let m = Measure::from_cmd(cmd, Measure::L2)?;
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        util::markout(ds, &mut mask, std::slice::from_ref(&idv));
        let rows = util::rows(&mask);
        let mut cols: Vec<Vec<f64>> = vars.iter().map(|v| util::col(ds, v).map(|c| rows.iter().map(|&i| c[i]).collect())).collect::<Result<_, _>>()?;
        let pick = |opt: &str| -> Result<Option<Vec<String>>, String> {
            match cmd.opts.iter().find(|(n, _)| n.eq_ignore_ascii_case(opt)) {
                Some((_, Some(a))) => Ok(Some(crate::varlist::expand(ds, a)?)),
                Some((_, None)) => Ok(Some(vars.clone())),
                None => Ok(None),
            }
        };
        let mut coding = Matrix::zeros(vars.len(), 2);
        for k in 0..vars.len() {
            coding[(k, 1)] = 1.0;
        }
        if let Some(u) = pick("std")? {
            for (k, v) in vars.iter().enumerate() {
                if u.contains(v) {
                    let mn = util::mean(&cols[k]);
                    let sd = util::variance(&cols[k]).sqrt();
                    coding[(k, 0)] = mn;
                    coding[(k, 1)] = sd;
                    cols[k] = cols[k].iter().map(|x| (x - mn) / sd).collect();
                }
            }
        }
        if let Some(u) = pick("unit")? {
            for (k, v) in vars.iter().enumerate() {
                if u.contains(v) {
                    let (mn, mx) = cols[k].iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, &x| (a.0.min(x), a.1.max(x)));
                    coding[(k, 0)] = mn;
                    coding[(k, 1)] = mx - mn;
                    cols[k] = cols[k].iter().map(|x| (x - mn) / (mx - mn)).collect();
                }
            }
        }
        let items: Vec<Vec<f64>> = (0..rows.len()).map(|t| cols.iter().map(|c| c[t]).collect()).collect();
        let mut d = crate::mv_cluster::pairwise(&items, m);
        if m.is_sim() {
            s2d(&mut d, cmd.opt("s2d", 3).unwrap_or("standard").trim());
        }
        let idvar = ds.get_var(&idv).ok_or_else(|| rc_err(111, format!("variable {idv} not found")))?;
        let ids: Vec<String> = rows.iter().map(|&i| util::cell_text(ds, idvar, i, false)).collect();
        let dtitle = format!("Dissimilarity: {}, computed on {} variable{}", m.name(), vars.len(), if vars.len() == 1 { "" } else { "s" });
        let vl = vars.join(" ");
        let coding = coding.with_names(vars.clone(), vec!["subtract".into(), "divide".into()]);
        let idc = idv.clone();
        run_mds(st, d, ids, &o, &dtitle, &m.name(), &|e| {
            e.set_str("id", idc.clone());
            e.set_str("varlist", vl.clone());
            e.set_mat("coding", coding.clone());
        }, Some(mask.clone()), cmd.has_opt("noplot", 3), cmd.has_opt("config", 4), cmd)
    }
}

pub struct Mdsmat;

impl CommandHandler for Mdsmat {
    fn name(&self) -> &'static str {
        "mdsmat"
    }

    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let name = cmd.main.split_whitespace().next().ok_or_else(|| rc_err(100, "matrix name required"))?.to_string();
        let mm = st.matrices.get(&name).cloned().ok_or_else(|| rc_err(111, format!("matrix {name} not found")))?;
        let o = mds_opts(cmd)?;
        let n = mm.rows;
        let mut d: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| if cmd.has_opt("force", 5) { 0.5 * (mm[(i, j)] + mm[(j, i)]) } else { mm[(i, j)] }).collect()).collect();
        if let Some(h) = cmd.opt("s2d", 3) {
            s2d(&mut d, h.trim());
        }
        let ids: Vec<String> = match cmd.opt("names", 2) {
            Some(s) => s.split_whitespace().map(String::from).collect(),
            None => if mm.rownames.len() == n { mm.rownames.clone() } else { mvutil::names("r", n) },
        };
        run_mds(st, d, ids, &o, &format!("Dissimilarity matrix: {name}"), &name, &|_| {}, None, cmd.has_opt("noplot", 3), cmd.has_opt("config", 4), cmd)
    }
}

pub struct Mdslong;

impl CommandHandler for Mdslong {
    fn name(&self) -> &'static str {
        "mdslong"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let dep = crate::varlist::expand(ds, &cmd.main)?.into_iter().next().ok_or_else(|| rc_err(100, "varlist required"))?;
        let idvars: Vec<String> = crate::varlist::expand(ds, cmd.opt("id", 2).ok_or_else(|| rc_err(198, "option id() required"))?)?;
        if idvars.len() != 2 {
            return Err(rc_err(198, "id() requires two variables"));
        }
        let o = mds_opts(cmd)?;
        let mut mask = util::touse(ds, st, cmd)?;
        let mut all = idvars.clone();
        all.push(dep.clone());
        util::markout(ds, &mut mask, &all);
        let rows = util::rows(&mask);
        let (v1, v2) = (ds.get_var(&idvars[0]).unwrap(), ds.get_var(&idvars[1]).unwrap());
        let y = util::col(ds, &dep)?;
        let mut labels: Vec<String> = Vec::new();
        let mut entries = Vec::new();
        for &i in &rows {
            let a = util::cell_text(ds, v1, i, false);
            let b = util::cell_text(ds, v2, i, false);
            for l in [&a, &b] {
                if !labels.contains(l) {
                    labels.push(l.clone());
                }
            }
            entries.push((a, b, y[i]));
        }
        labels.sort_by(|a, b| match (a.parse::<f64>(), b.parse::<f64>()) {
            (Ok(x), Ok(z)) => x.total_cmp(&z),
            _ => a.cmp(b),
        });
        let n = labels.len();
        let sim = cmd.opt("s2d", 3).is_some();
        let mut d = vec![vec![SYSMISS; n]; n];
        for i in 0..n {
            d[i][i] = if sim { 1.0 } else { 0.0 };
        }
        for (a, b, v) in entries {
            let i = labels.iter().position(|l| *l == a).unwrap();
            let j = labels.iter().position(|l| *l == b).unwrap();
            if i == j {
                if sim {
                    d[i][i] = v;
                }
                continue;
            }
            if !is_missing(d[i][j]) && (d[i][j] - v).abs() > 1e-12 && !cmd.has_opt("force", 5) {
                return Err(rc_err(198, "proximity data not symmetric; specify force"));
            }
            let nv = if is_missing(d[i][j]) { v } else { 0.5 * (d[i][j] + v) };
            d[i][j] = nv;
            d[j][i] = nv;
        }
        if d.iter().flatten().any(|v| is_missing(*v)) {
            return Err(rc_err(198, "proximities missing for some pairs"));
        }
        if let Some(h) = cmd.opt("s2d", 3) {
            s2d(&mut d, h.trim());
        }
        run_mds(st, d, labels, &o, &format!("Dissimilarity variable: {dep} in long format"), &dep, &|_| {}, Some(mask.clone()), cmd.has_opt("noplot", 3), cmd.has_opt("config", 4), cmd)
    }
}

/// estat after mds: config, stress.
pub fn mds_estat(st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    if st.e.macros.get("cmd").map(String::as_str) != Some("mds") {
        return None;
    }
    let y = st.e.matrices.get("Y").cloned()?;
    let d = st.e.matrices.get("D").cloned()?;
    let n = y.rows;
    match sub {
        "config" => {
            let mut out = vec![String::new(), format!("Configuration in {}-dimensional Euclidean space (principal normalization)", y.cols), String::new()];
            out.extend(mvutil::matlist(&y, "", cmd.opt("format", 3).unwrap_or("%9.4f"), 9, false, 4));
            st.begin_r().set_mat("Y", y);
            Some(Ok(out.join("\n")))
        }
        "stress" | "correlations" | "pairwise" => {
            let rows: Vec<Vec<f64>> = (0..n).map(|i| y.row(i).to_vec()).collect();
            let b = dist_of(&rows);
            let mut out = vec![String::new()];
            let mut per = Vec::new();
            let (mut num, mut den) = (0.0, 0.0);
            for i in 0..n {
                let (mut a, mut c) = (0.0, 0.0);
                for j in 0..n {
                    if i != j {
                        a += (b[i][j] - d[(i, j)]).powi(2);
                        c += b[i][j] * b[i][j];
                    }
                }
                per.push((a / c).sqrt());
                num += a;
                den += c;
            }
            let total = (num / den).sqrt();
            if sub == "stress" {
                out.push("Kruskal stress (stress-1) between dissimilarities and distances".into());
                out.push(String::new());
                let mut bx = mvutil::Boxed::new("", vec![vec![("Kruskal".into(), 10)]]);
                for i in 0..n {
                    bx.row(&y.rownames[i], vec![format!("{:.4}", per[i])]);
                }
                bx.seps.push(n - 1);
                bx.row("Total", vec![format!("{total:.4}")]);
                out.extend(bx.render());
                let r = st.begin_r();
                r.set("stress", total);
                r.set_mat("S", Matrix::col_vector(&per).with_names(y.rownames.clone(), vec!["Kruskal".into()]));
            } else {
                // Pearson and Spearman correlations of dissimilarities and distances per object
                out.push("Correlations of dissimilarities and fitted distances".into());
                out.push(String::new());
                let mut bx = mvutil::Boxed::new("", vec![vec![("N".into(), 6), ("Pearson".into(), 10), ("Spearman".into(), 10)]]);
                let corr = |a: &[f64], c: &[f64]| -> f64 {
                    let (ma, mc) = (util::mean(a), util::mean(c));
                    let sac: f64 = a.iter().zip(c).map(|(x, z)| (x - ma) * (z - mc)).sum();
                    sac / (a.iter().map(|x| (x - ma).powi(2)).sum::<f64>() * c.iter().map(|z| (z - mc).powi(2)).sum::<f64>()).sqrt()
                };
                let mut rows_out = Vec::new();
                for i in 0..n {
                    let a: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| d[(i, j)]).collect();
                    let c: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| b[i][j]).collect();
                    let pr = corr(&a, &c);
                    let sr = corr(&crate::correlate::ranks(&a), &crate::correlate::ranks(&c));
                    bx.row(&y.rownames[i], vec![(n - 1).to_string(), format!("{pr:.4}"), format!("{sr:.4}")]);
                    rows_out.push(vec![(n - 1) as f64, pr, sr]);
                }
                out.extend(bx.render());
                st.begin_r().set_mat("R", Matrix::from_rows(&rows_out).with_names(y.rownames.clone(), vec!["N".into(), "Pearson".into(), "Spearman".into()]));
            }
            Some(Ok(out.join("\n")))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// correspondence analysis
// ---------------------------------------------------------------------------

/// SVD of the standardized residuals of a two-way table.
struct CaFit {
    n: f64,
    r: Vec<f64>,
    c: Vec<f64>,
    sv: Vec<f64>,
    /// standard coordinates
    a: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
}

fn ca_fit(t: &[Vec<f64>]) -> CaFit {
    let nr = t.len();
    let nc = t[0].len();
    let n: f64 = t.iter().flatten().sum();
    let r: Vec<f64> = t.iter().map(|row| row.iter().sum::<f64>() / n).collect();
    let c: Vec<f64> = (0..nc).map(|j| t.iter().map(|row| row[j]).sum::<f64>() / n).collect();
    let mut s = Matrix::zeros(nr, nc);
    for i in 0..nr {
        for j in 0..nc {
            s[(i, j)] = (t[i][j] / n - r[i] * c[j]) / (r[i] * c[j]).sqrt();
        }
    }
    let sts = s.transpose().mul(&s).unwrap();
    let (vals, v) = sts.sym_eigen();
    let k = nr.min(nc) - 1;
    let mut sv = Vec::new();
    let mut a = vec![vec![0.0; k]; nr];
    let mut b = vec![vec![0.0; k]; nc];
    for d in 0..k {
        let sg = vals[d].max(0.0).sqrt();
        sv.push(sg);
        let mut u: Vec<f64> = (0..nr).map(|i| (0..nc).map(|j| s[(i, j)] * v[(j, d)]).sum::<f64>() / sg.max(1e-300)).collect();
        let mut vv: Vec<f64> = (0..nc).map(|j| v[(j, d)]).collect();
        // orientation: positive coordinate sum weighted... first row category with largest |u| positive
        let s0: f64 = u.iter().sum::<f64>() + vv.iter().sum::<f64>();
        let _ = s0;
        let big = (0..nc).max_by(|&x, &y| vv[x].abs().total_cmp(&vv[y].abs())).unwrap_or(0);
        if vv[big] < 0.0 {
            u.iter_mut().for_each(|x| *x = -*x);
            vv.iter_mut().for_each(|x| *x = -*x);
        }
        for i in 0..nr {
            a[i][d] = u[i] / r[i].sqrt();
        }
        for j in 0..nc {
            b[j][d] = vv[j] / c[j].sqrt();
        }
    }
    CaFit { n, r, c, sv, a, b }
}

fn norm_alpha(cmd: &Command) -> Result<(f64, String), String> {
    let s = cmd.opt("normalize", 4).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "symmetric".into());
    Ok(match s.as_str() {
        x if x.starts_with("sym") => (0.5, "symmetric".into()),
        x if x.starts_with("stand") => (0.0, "standard".into()),
        x if x.starts_with("prin") => (-1.0, "principal".into()),
        x if x.starts_with("row") => (1.0, "row".into()),
        x if x.starts_with("col") => (0.0, "column".into()),
        x => (x.parse::<f64>().map_err(|_| rc_err(198, format!("invalid normalize({x})")))?, x.to_string()),
    })
}

/// Point statistics table for row or column categories.
fn ca_stats(names: &[String], mass: &[f64], std_: &[Vec<f64>], sv: &[f64], coord_pow: f64, dims: usize, total: f64) -> (Matrix, Vec<Vec<String>>) {
    let k = sv.len();
    let cols = 3 + 3 * dims;
    let mut m = Matrix::zeros(names.len(), cols);
    let mut cells = Vec::new();
    for i in 0..names.len() {
        let inert: f64 = (0..k).map(|d| mass[i] * (std_[i][d] * sv[d]).powi(2)).sum();
        let mut q = 0.0;
        let mut row = vec![0.0; cols];
        row[0] = mass[i];
        row[2] = inert / total;
        for d in 0..dims {
            let princ = std_[i][d] * sv[d];
            let sq = if inert > 0.0 { mass[i] * princ * princ / inert } else { 0.0 };
            q += sq;
            row[3 + 3 * d] = std_[i][d] * sv[d].powf(coord_pow);
            row[4 + 3 * d] = sq;
            row[5 + 3 * d] = mass[i] * std_[i][d] * std_[i][d];
        }
        row[1] = q;
        for (j, v) in row.iter().enumerate() {
            m[(i, j)] = *v;
        }
        cells.push(row.iter().map(|v| format!("{v:.3}")).collect());
    }
    (m, cells)
}

fn ca_output(fit: &CaFit, rnames: &[String], cnames: &[String], rvar: &str, cvar: &str, cmd: &Command, st: &mut Store, dims_req: usize) -> Result<String, String> {
    let k = fit.sv.len();
    let dims = dims_req.min(k).max(1);
    let (alpha, normname) = norm_alpha(cmd)?;
    let (rp, cp) = if alpha < 0.0 { (1.0, 1.0) } else { (alpha, 1.0 - alpha) };
    let inertia: f64 = fit.sv.iter().map(|s| s * s).sum();
    let chi2 = fit.n * inertia;
    let df = ((rnames.len() - 1) * (cnames.len() - 1)) as f64;
    let pv = dist::chi2tail(df, chi2);
    let expl: f64 = fit.sv[..dims].iter().map(|s| s * s).sum::<f64>() / inertia * 100.0;
    let mut out = vec![String::new()];
    out.push(mvutil::hdr("Correspondence analysis", "Number of obs", &util::commas(fit.n as usize)));
    out.push(mvutil::hdr("", &format!("Pearson chi2({})", df), &format!("{chi2:.2}")));
    out.push(mvutil::hdr("", "Prob > chi2", &format!("{pv:.4}")));
    out.push(mvutil::hdr("", "Total inertia", &format!("{inertia:.4}")));
    out.push(mvutil::hdr(&format!("    {} active rows", rnames.len()), "Number of dim.", &dims.to_string()));
    out.push(mvutil::hdr(&format!("    {} active columns", cnames.len()), "Expl. inertia (%)", &format!("{expl:.2}")));
    out.push(String::new());
    out.push(format!("    {}", "-".repeat(70)));
    out.push(format!("    {:>14} | {:>11} {:>11} {:>10} {:>9} {:>9}", "", "Singular", "Principal", "", "", "Cumul."));
    out.push(format!("    {:>14} | {:>11} {:>11} {:>10} {:>9} {:>9}", "Dimension", "value", "inertia", "chi2", "Percent", "percent"));
    out.push(format!("    {}+{}", "-".repeat(15), "-".repeat(54)));
    let mut cum = 0.0;
    for d in 0..k {
        let pi = fit.sv[d] * fit.sv[d];
        cum += pi / inertia * 100.0;
        out.push(format!("    {:>14} | {:>11} {:>11} {:>10.2} {:>9.2} {:>9.2}", format!("Dim {}", d + 1), sig(fit.sv[d], 7), sig(pi, 7), fit.n * pi, pi / inertia * 100.0, cum));
    }
    out.push(format!("    {}+{}", "-".repeat(15), "-".repeat(54)));
    out.push(format!("    {:>14} | {:>11} {:>11} {:>10.2} {:>9}", "Total", "", sig(inertia, 7), chi2, 100));
    out.push(format!("    {}", "-".repeat(70)));
    let (gr, rcells) = ca_stats(rnames, &fit.r, &fit.a, &fit.sv, rp, dims, inertia);
    let (gc, ccells) = ca_stats(cnames, &fit.c, &fit.b, &fit.sv, cp, dims, inertia);
    if !cmd.has_opt("compact", 4) || true {
        out.push(String::new());
        out.push(format!("Statistics for row and column categories in {normname} normalization"));
        out.push(String::new());
        for d0 in std::iter::once(None).chain((0..dims).map(Some)) {
            let groups: Vec<Vec<(String, usize)>> = match d0 {
                None => vec![vec![("Mass".into(), 9), ("Quality".into(), 9), ("%inert".into(), 9)], vec![("Coord".into(), 9), ("Sqcorr".into(), 9), ("Contrib".into(), 9)]],
                Some(_) => vec![vec![("Coord".into(), 9), ("Sqcorr".into(), 9), ("Contrib".into(), 9)]],
            };
            if d0 == Some(0) {
                continue;
            }
            let mut bx = mvutil::Boxed::new("Categories", groups);
            bx.stub_width = 14;
            bx.row(rvar, vec![]);
            let pick = |cells: &Vec<String>| -> Vec<String> {
                match d0 {
                    None => cells[0..6].to_vec(),
                    Some(d) => cells[3 + 3 * d..6 + 3 * d].to_vec(),
                }
            };
            for (i, nm) in rnames.iter().enumerate() {
                bx.row(&format!("  {nm}"), pick(&rcells[i]));
            }
            bx.row(cvar, vec![]);
            for (j, nm) in cnames.iter().enumerate() {
                bx.row(&format!("  {nm}"), pick(&ccells[j]));
            }
            let lines = bx.render();
            let title = match d0 {
                None => format!("    {:16}{:^30}{:^30}", "", "Overall", "Dimension_1"),
                Some(d) => format!("    {:16}{:^30}", "", format!("Dimension_{}", d + 1)),
            };
            out.push(title);
            out.extend(lines);
            out.push(String::new());
        }
    }
    let mut rc = Matrix::zeros(rnames.len(), dims);
    let mut cc = Matrix::zeros(cnames.len(), dims);
    for d in 0..dims {
        for i in 0..rnames.len() {
            rc[(i, d)] = fit.a[i][d] * fit.sv[d].powf(rp);
        }
        for j in 0..cnames.len() {
            cc[(j, d)] = fit.b[j][d] * fit.sv[d].powf(cp);
        }
    }
    let dn = mvutil::names("dim", dims);
    let mut statn = vec!["mass".to_string(), "quality".into(), "inertia".into()];
    for d in 1..=dims {
        statn.push(format!("dim{d}"));
        statn.push(format!("cos{d}"));
        statn.push(format!("contr{d}"));
    }
    let e = st.begin_e();
    e.set("N", fit.n);
    e.set("f", dims as f64);
    e.set("inertia", inertia);
    e.set("pinertia", fit.sv[..dims].iter().map(|s| s * s).sum());
    e.set("chi2", chi2);
    e.set("df", df);
    e.set("p_chi2", pv);
    e.set_str("cmd", "ca");
    e.set_str("norm", normname);
    e.set_str("Rname", rvar);
    e.set_str("Cname", cvar);
    e.set_str("properties", "nob noV eigen");
    e.set_str("estat_cmd", "ca_estat");
    e.set_str("predict", "ca_p");
    e.set_mat("Sv", Matrix::row_vector(&fit.sv).with_names(vec!["r1".into()], mvutil::names("dim", k)));
    e.set_mat("TR", rc.with_names(rnames.to_vec(), dn.clone()));
    e.set_mat("TC", cc.with_names(cnames.to_vec(), dn));
    e.set_mat("GSR", gr.with_names(rnames.to_vec(), statn.clone()));
    e.set_mat("GSC", gc.with_names(cnames.to_vec(), statn));
    e.set_mat("rowmass", Matrix::col_vector(&fit.r).with_names(rnames.to_vec(), vec!["mass".into()]));
    e.set_mat("colmass", Matrix::col_vector(&fit.c).with_names(cnames.to_vec(), vec!["mass".into()]));
    let _ = alpha;
    if cmd.has_opt("plot", 4) {
        let _ = ca_plot(st, cmd);
    }
    Ok(out.join("\n"))
}

fn ca_plot(st: &mut Store, cmd: &Command) -> Result<String, String> {
    let tr = st.e.matrices.get("TR").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let tc = st.e.matrices.get("TC").cloned().unwrap();
    let mut g = Graph { title: "Correspondence analysis biplot".into(), xtitle: "Dimension 1".into(), ytitle: "Dimension 2".into(), legend: true, ..Default::default() };
    let pt = |m: &Matrix| -> Vec<(f64, f64)> { (0..m.rows).map(|i| (m[(i, 0)], if m.cols > 1 { m[(i, 1)] } else { 0.0 })).collect() };
    g.layers.push(Layer { kind: LayerKind::Scatter, points: pt(&tr), mlabels: tr.rownames.clone(), label: st.e.macros.get("Rname").cloned().unwrap_or_default(), ..Default::default() });
    g.layers.push(Layer { kind: LayerKind::Scatter, points: pt(&tc), mlabels: tc.rownames.clone(), label: st.e.macros.get("Cname").cloned().unwrap_or_default(), hollow: true, ..Default::default() });
    g.command = cmd.raw.clone();
    let mut c = cmd.clone();
    c.opts.retain(|(n, _)| ["title", "name", "saving"].iter().any(|k| n.eq_ignore_ascii_case(k)));
    finish(&mut g, &c, st)
}

pub struct Cabiplot;
impl CommandHandler for Cabiplot {
    fn name(&self) -> &'static str {
        "cabiplot"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        ca_plot(st, cmd)
    }
}

/// Distinct categories of a variable over rows, with display labels.
fn categories(ds: &DataSet, v: &str, rows: &[usize]) -> Result<(Vec<f64>, Vec<String>), String> {
    let var = ds.get_var(v).ok_or_else(|| rc_err(111, format!("variable {v} not found")))?;
    let c = util::col(ds, v)?;
    let mut vals: Vec<f64> = rows.iter().map(|&i| c[i]).collect();
    vals.sort_by(|a, b| a.total_cmp(b));
    vals.dedup();
    let labels = vals
        .iter()
        .map(|&x| {
            let i = rows.iter().copied().find(|&i| c[i] == x).unwrap();
            util::cell_text(ds, var, i, true)
        })
        .collect();
    Ok((vals, labels))
}

pub struct Ca;

impl CommandHandler for Ca {
    fn name(&self) -> &'static str {
        "ca"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = crate::varlist::expand(ds, &cmd.main)?;
        if vars.len() != 2 {
            return Err(rc_err(198, "ca requires two variables"));
        }
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        let w = util::weights(ds, st, cmd, &mut mask, "fai")?;
        let rows = util::rows(&mask);
        let (rv, rl) = categories(ds, &vars[0], &rows)?;
        let (cv, cl) = categories(ds, &vars[1], &rows)?;
        let rc = util::col(ds, &vars[0])?;
        let cc = util::col(ds, &vars[1])?;
        let mut t = vec![vec![0.0; cv.len()]; rv.len()];
        for &i in &rows {
            let a = rv.iter().position(|&x| x == rc[i]).unwrap();
            let b = cv.iter().position(|&x| x == cc[i]).unwrap();
            t[a][b] += w.as_ref().map(|w| w.w[i]).unwrap_or(1.0);
        }
        if rv.len() < 2 || cv.len() < 2 {
            return Err(rc_err(198, "ca requires at least two categories per variable"));
        }
        let fit = ca_fit(&t);
        let dims = mvutil::opt_usize(cmd, "dimensions", 3)?.unwrap_or(2);
        let out = ca_output(&fit, &rl, &cl, &vars[0], &vars[1], cmd, st, dims)?;
        st.e.sample = Some(mask);
        Ok(out)
    }
}

pub struct Camat;

impl CommandHandler for Camat {
    fn name(&self) -> &'static str {
        "camat"
    }

    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let name = cmd.main.split_whitespace().next().ok_or_else(|| rc_err(100, "matrix name required"))?;
        let m = st.matrices.get(name).cloned().ok_or_else(|| rc_err(111, format!("matrix {name} not found")))?;
        if m.data.iter().any(|v| *v < 0.0 || is_missing(*v)) {
            return Err(rc_err(198, "matrix must contain nonnegative values"));
        }
        let t: Vec<Vec<f64>> = (0..m.rows).map(|i| m.row(i).to_vec()).collect();
        let fit = ca_fit(&t);
        let rl = if m.rownames.len() == m.rows { m.rownames.clone() } else { mvutil::names("r", m.rows) };
        let cl = if m.colnames.len() == m.cols { m.colnames.clone() } else { mvutil::names("c", m.cols) };
        let dims = mvutil::opt_usize(cmd, "dimensions", 3)?.unwrap_or(2);
        let rn = cmd.opt("rowname", 4).unwrap_or("rows").trim().to_string();
        let cn = cmd.opt("colname", 4).unwrap_or("columns").trim().to_string();
        ca_output(&fit, &rl, &cl, &rn, &cn, cmd, st, dims)
    }
}

/// `estat` after ca: table (fitted/expected), coordinates.
pub fn ca_estat(st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    if st.e.macros.get("cmd").map(String::as_str) != Some("ca") {
        return None;
    }
    match sub {
        "coordinates" | "coord" | "coordinate" => {
            let tr = st.e.matrices.get("TR").cloned()?;
            let tc = st.e.matrices.get("TC").cloned()?;
            let mut out = vec![String::new(), format!("Row and column coordinates in {} normalization", st.e.macros.get("norm").cloned().unwrap_or_default()), String::new()];
            out.extend(mvutil::matlist(&tr, st.e.macros.get("Rname").map(String::as_str).unwrap_or("rows"), cmd.opt("format", 3).unwrap_or("%9.4f"), 9, false, 4));
            out.push(String::new());
            out.extend(mvutil::matlist(&tc, st.e.macros.get("Cname").map(String::as_str).unwrap_or("columns"), cmd.opt("format", 3).unwrap_or("%9.4f"), 9, false, 4));
            let r = st.begin_r();
            r.set_mat("Rcoords", tr);
            r.set_mat("Ccoords", tc);
            Some(Ok(out.join("\n")))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// multiple correspondence analysis
// ---------------------------------------------------------------------------

pub struct Mca;

impl CommandHandler for Mca {
    fn name(&self) -> &'static str {
        "mca"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = crate::varlist::expand(ds, &cmd.main)?;
        if vars.len() < 2 {
            return Err(rc_err(102, "too few variables specified"));
        }
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        let rows = util::rows(&mask);
        let method = cmd.opt("method", 4).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "burt".into());
        let indicator = method.starts_with("ind");
        let adjust = !indicator && !cmd.has_opt("noadjust", 5);
        // indicator matrix
        let mut cats: Vec<(String, Vec<f64>, Vec<String>)> = Vec::new();
        for v in &vars {
            let (vals, labels) = categories(ds, v, &rows)?;
            cats.push((v.clone(), vals, labels));
        }
        let q = vars.len();
        let jtot: usize = cats.iter().map(|c| c.1.len()).sum();
        let n = rows.len();
        let mut z = vec![vec![0.0; jtot]; n];
        let mut off = 0;
        for (v, vals, _) in &cats {
            let c = util::col(ds, v)?;
            for (t, &i) in rows.iter().enumerate() {
                let k = vals.iter().position(|&x| x == c[i]).unwrap();
                z[t][off + k] = 1.0;
            }
            off += vals.len();
        }
        let fit = ca_fit(&z);
        let lam: Vec<f64> = fit.sv.iter().map(|s| s * s).collect();
        let qf = q as f64;
        let (inert, total, title): (Vec<f64>, f64, &str) = if indicator {
            (lam.iter().copied().take(jtot - q).filter(|&l| l > 1e-12).collect(), (jtot as f64 - qf) / qf, "Indicator matrix")
        } else if adjust {
            let adj: Vec<f64> = lam.iter().filter(|&&l| l > 1.0 / qf).map(|&l| (qf / (qf - 1.0)).powi(2) * (l - 1.0 / qf).powi(2)).collect();
            let burt_total: f64 = lam.iter().map(|l| l * l).sum();
            let tot = qf / (qf - 1.0) * (burt_total - (jtot as f64 - qf) / (qf * qf));
            (adj, tot, "Burt/adjusted inertias")
        } else {
            let b: Vec<f64> = lam.iter().map(|l| l * l).collect();
            let tot = b.iter().sum();
            (b, tot, "Burt/unadjusted inertias")
        };
        let dims = mvutil::opt_usize(cmd, "dimensions", 3)?.unwrap_or(2).min(inert.len()).max(1);
        let (alpha, normname) = {
            let s = cmd.opt("normalize", 4).map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "standard".into());
            if s.starts_with("prin") { (1.0, "principal") } else { (0.0, "standard") }
        };
        let mut out = vec![String::new()];
        out.push(mvutil::hdr("Multiple/joint correspondence analysis", "Number of obs", &util::commas(n)));
        out.push(mvutil::hdr("", "Total inertia", &sig(total, 8)));
        out.push(mvutil::hdr(&format!("    Method: {title}"), "Number of axes", &dims.to_string()));
        out.push(String::new());
        out.push(format!("    {}", "-".repeat(50)));
        out.push(format!("    {:>14} | {:>11} {:>10} {:>10}", "", "Principal", "", "Cumul."));
        out.push(format!("    {:>14} | {:>11} {:>10} {:>10}", "Dimension", "inertia", "Percent", "percent"));
        out.push(format!("    {}+{}", "-".repeat(15), "-".repeat(34)));
        let mut cum = 0.0;
        for (d, v) in inert.iter().enumerate() {
            cum += v / total * 100.0;
            out.push(format!("    {:>14} | {:>11} {:>10.2} {:>10.2}", format!("Dim {}", d + 1), sig(*v, 7), v / total * 100.0, cum));
        }
        out.push(format!("    {}+{}", "-".repeat(15), "-".repeat(34)));
        out.push(format!("    {:>14} | {:>11} {:>10.2}", "Total", sig(total, 7), 100.0));
        out.push(format!("    {}", "-".repeat(50)));
        // column coordinates
        let mut names = Vec::new();
        let mut coords = Matrix::zeros(jtot, dims);
        for (_, _, labels) in &cats {
            names.extend(labels.iter().cloned());
        }
        for j in 0..jtot {
            for d in 0..dims {
                coords[(j, d)] = fit.b[j][d] * if alpha > 0.0 { inert[d].sqrt() } else { 1.0 };
            }
        }
        out.push(String::new());
        out.push(format!("Coordinates of column categories in {normname} normalization"));
        out.push(String::new());
        let dn = mvutil::names("dim", dims);
        let mut bx = mvutil::Boxed::new("Categories", vec![std::iter::once(("Mass".to_string(), 9)).chain(dn.iter().map(|d| (d.clone(), 9))).collect()]);
        bx.stub_width = 14;
        let mut j = 0;
        for (v, _, labels) in &cats {
            bx.row(v, vec![]);
            for l in labels {
                let mut cells = vec![format!("{:.3}", fit.c[j])];
                cells.extend((0..dims).map(|d| format!("{:.3}", coords[(j, d)])));
                bx.row(&format!("  {l}"), cells);
                j += 1;
            }
        }
        out.extend(bx.render());
        let e = st.begin_e();
        e.set("N", n as f64);
        e.set("f", dims as f64);
        e.set("inertia", total);
        e.set_str("cmd", "mca");
        e.set_str("method", if indicator { "indicator" } else { "burt" });
        e.set_str("norm", normname);
        e.set_str("names", vars.join(" "));
        e.set_str("properties", "nob noV eigen");
        e.set_mat("inertia_e", Matrix::row_vector(&inert).with_names(vec!["r1".into()], mvutil::names("dim", inert.len())));
        e.set_mat("cGS", coords.with_names(names, dn));
        e.sample = Some(mask);
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// biplot
// ---------------------------------------------------------------------------

pub struct Biplot;

impl CommandHandler for Biplot {
    fn name(&self) -> &'static str {
        "biplot"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = crate::varlist::expand(ds, &cmd.main)?;
        if vars.len() < 2 {
            return Err(rc_err(102, "too few variables specified"));
        }
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        let rows = util::rows(&mask);
        let n = rows.len();
        let p = vars.len();
        let std_ = cmd.has_opt("std", 3);
        let alpha = util::opt_num(cmd, "alpha", 2)?.unwrap_or(0.5);
        let stretch = util::opt_num(cmd, "stretch", 3)?.unwrap_or(1.0);
        let dimsel: Vec<usize> = match cmd.opt("dim", 3) {
            Some(s) => s.split_whitespace().filter_map(|w| w.parse::<usize>().ok()).map(|v| v.saturating_sub(1)).collect(),
            None => vec![0, 1],
        };
        let mut x = Matrix::zeros(n, p);
        for (k, v) in vars.iter().enumerate() {
            let c = util::col(ds, v)?;
            let vals: Vec<f64> = rows.iter().map(|&i| c[i]).collect();
            let m = util::mean(&vals);
            let sd = if std_ { util::variance(&vals).sqrt() } else { 1.0 };
            for t in 0..n {
                x[(t, k)] = (vals[t] - m) / sd;
            }
        }
        let xtx = x.transpose().mul(&x).unwrap();
        let (vals, v) = mvutil::sym_eig(&xtx);
        let sv: Vec<f64> = vals.iter().map(|l| l.max(0.0).sqrt()).collect();
        let tot: f64 = vals.iter().sum();
        let (d1, d2) = (dimsel[0], *dimsel.get(1).unwrap_or(&1));
        let rowc = |t: usize, d: usize| -> f64 { (0..p).map(|k| x[(t, k)] * v[(k, d)]).sum::<f64>() / sv[d] * sv[d].powf(alpha) };
        let colc = |k: usize, d: usize| -> f64 { v[(k, d)] * sv[d].powf(1.0 - alpha) * stretch };
        let mut out = vec![String::new()];
        out.push(format!("Biplot of {} observations and {} variables", n, p));
        out.push(String::new());
        out.push(format!("    Explained variance by component {} = {:>7.4}", d1 + 1, vals[d1] / tot));
        out.push(format!("    Explained variance by component {} = {:>7.4}", d2 + 1, vals[d2] / tot));
        out.push(format!("              Total explained variance = {:>7.4}", (vals[d1] + vals[d2]) / tot));
        let mut rm = Matrix::zeros(n, 2);
        let mut cm = Matrix::zeros(p, 2);
        for t in 0..n {
            rm[(t, 0)] = rowc(t, d1);
            rm[(t, 1)] = rowc(t, d2);
        }
        for k in 0..p {
            cm[(k, 0)] = colc(k, d1);
            cm[(k, 1)] = colc(k, d2);
        }
        let dn = vec![format!("dim{}", d1 + 1), format!("dim{}", d2 + 1)];
        if cmd.has_opt("table", 3) {
            out.push(String::new());
            out.push("Biplot coordinates".into());
            out.push(String::new());
            out.extend(mvutil::matlist(&cm.clone().with_names(vars.clone(), dn.clone()), "Variable", "%9.4f", 9, false, 4));
        }
        if let Some(g) = cmd.opt("generate", 3) {
            let w: Vec<&str> = g.split_whitespace().collect();
            if w.len() == 2 {
                let n0 = ds.n_obs;
                mvutil::put_var(ds, st, w[0], None, n0, &rows, &(0..n).map(|t| rm[(t, 0)]).collect::<Vec<_>>(), "")?;
                mvutil::put_var(ds, st, w[1], None, n0, &rows, &(0..n).map(|t| rm[(t, 1)]).collect::<Vec<_>>(), "")?;
            }
        }
        let mut g = Graph { title: "Biplot".into(), xtitle: format!("Dimension {}", d1 + 1), ytitle: format!("Dimension {}", d2 + 1), legend: true, ..Default::default() };
        if !cmd.has_opt("norow", 5) {
            g.layers.push(Layer { kind: LayerKind::Scatter, points: (0..n).map(|t| (rm[(t, 0)], rm[(t, 1)])).collect(), label: "Observations".into(), ..Default::default() });
        }
        if !cmd.has_opt("nocolumn", 5) {
            g.layers.push(Layer { kind: LayerKind::Pc(PcStyle::Arrow), segments: (0..p).map(|k| [0.0, 0.0, cm[(k, 1)], cm[(k, 0)]]).collect(), label: "Variables".into(), ..Default::default() });
            g.layers.push(Layer { kind: LayerKind::Scatter, points: (0..p).map(|k| (cm[(k, 0)], cm[(k, 1)])).collect(), mlabels: vars.clone(), symbol: "none".into(), ..Default::default() });
        }
        g.command = cmd.raw.clone();
        let mut c = cmd.clone();
        c.opts.retain(|(n, _)| ["title", "name", "saving", "xtitle", "ytitle"].iter().any(|k| n.eq_ignore_ascii_case(k)));
        let _ = finish(&mut g, &c, st)?;
        let r = st.begin_r();
        r.set("rho1", vals[d1] / tot);
        r.set("rho2", vals[d2] / tot);
        r.set("rho", (vals[d1] + vals[d2]) / tot);
        r.set_mat("U", rm.with_names(rows.iter().map(|i| format!("obs{}", i + 1)).collect(), dn.clone()));
        r.set_mat("V", cm.with_names(vars.clone(), dn));
        r.set_mat("Vstretch", Matrix::filled(1, 1, stretch));
        Ok(out.join("\n"))
    }
}
