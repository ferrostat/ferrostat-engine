//! Orthogonal and oblique rotations of loading matrices: the engine behind
//! `rotate` (after pca/factor) and `rotatemat`.
//!
//! Varimax uses the classic Kaiser iteration; the other criteria use the
//! gradient projection algorithms of Jennrich (orthogonal and oblique);
//! promax is a Procrustean oblique rotation toward a power of varimax.

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::mvutil::{self, polar};
use crate::store::{Results, Store};
use crate::util;

#[derive(Debug, Clone)]
pub enum Crit {
    Varimax,
    Vgpf,
    Quartimax,
    Equamax,
    Parsimax,
    Entropy,
    Tandem1,
    Tandem2,
    Promax(f64),
    Oblimin(f64),
    Cf(f64),
    Bentler,
    Oblimax,
    Quartimin,
    Target(Matrix),
    Partial(Matrix, Matrix),
}

impl Crit {
    pub fn name(&self) -> &'static str {
        match self {
            Crit::Varimax => "varimax",
            Crit::Vgpf => "vgpf",
            Crit::Quartimax => "quartimax",
            Crit::Equamax => "equamax",
            Crit::Parsimax => "parsimax",
            Crit::Entropy => "entropy",
            Crit::Tandem1 => "tandem1",
            Crit::Tandem2 => "tandem2",
            Crit::Promax(_) => "promax",
            Crit::Oblimin(_) => "oblimin",
            Crit::Cf(_) => "cf",
            Crit::Bentler => "bentler",
            Crit::Oblimax => "oblimax",
            Crit::Quartimin => "quartimin",
            Crit::Target(_) => "target",
            Crit::Partial(..) => "partial",
        }
    }
    fn orth_only(&self) -> bool {
        matches!(self, Crit::Varimax | Crit::Vgpf | Crit::Quartimax | Crit::Equamax | Crit::Parsimax | Crit::Entropy | Crit::Tandem1 | Crit::Tandem2)
    }
    /// Title used by `estat common`: oblimin(0), promax(3), ...
    pub fn ctitle(&self) -> String {
        match self {
            Crit::Promax(p) => format!("promax({})", crate::format::fmt_g(*p, 8)),
            Crit::Oblimin(g) => format!("oblimin({})", crate::format::fmt_g(*g, 8)),
            Crit::Cf(k) => format!("cf({})", crate::format::fmt_g(*k, 8)),
            c => c.name().to_string(),
        }
    }
}

pub struct RotOpts {
    pub crit: Crit,
    pub oblique: bool,
    pub normalize: bool,
}

/// Parse the rotation options of rotate/rotatemat.
pub fn parse_opts(st: &Store, cmd: &Command) -> Result<RotOpts, String> {
    let mut crit = Crit::Varimax;
    let mut oblique = cmd.has_opt("oblique", 3);
    for (n, a) in &cmd.opts {
        let n = n.to_lowercase();
        let num = |d: f64| -> Result<f64, String> {
            match a {
                Some(s) if !s.trim().is_empty() => {
                    let e = ferrostat_parser::parser::parse_expression(s).map_err(|e| rc_err(198, e.to_string()))?;
                    crate::eval::Env::with_store(&DataSet::new(), st).eval(&e, 0)?.num()
                }
                _ => Ok(d),
            }
        };
        let mat = |s: &Option<String>| -> Result<Matrix, String> {
            let nm = s.as_deref().unwrap_or("").trim().to_string();
            st.matrices.get(&nm).cloned().ok_or_else(|| rc_err(111, format!("matrix {nm} not found")))
        };
        crit = match n.as_str() {
            "varimax" | "varim" | "vari" => Crit::Varimax,
            "vgpf" => Crit::Vgpf,
            "quartimax" | "quartima" | "quartim" if n != "quartimin" => Crit::Quartimax,
            "equamax" | "equa" => Crit::Equamax,
            "parsimax" | "pars" => Crit::Parsimax,
            "entropy" | "entr" => Crit::Entropy,
            "tandem1" => Crit::Tandem1,
            "tandem2" => Crit::Tandem2,
            "promax" | "prom" => {
                oblique = true;
                let p = num(3.0)?;
                if p <= 1.0 {
                    return Err(rc_err(198, "promax power must be greater than 1"));
                }
                Crit::Promax(p)
            }
            "oblimin" | "oblimi" => Crit::Oblimin(num(0.0)?),
            "cf" => Crit::Cf(num(0.0)?),
            "bentler" | "bent" => Crit::Bentler,
            "oblimax" => Crit::Oblimax,
            "quartimin" => Crit::Quartimin,
            "target" | "targ" => Crit::Target(mat(a)?),
            "partial" | "part" => {
                let words: Vec<String> = a.as_deref().unwrap_or("").split_whitespace().map(String::from).collect();
                if words.len() != 2 {
                    return Err(rc_err(198, "partial() requires two matrices"));
                }
                Crit::Partial(mat(&Some(words[0].clone()))?, mat(&Some(words[1].clone()))?)
            }
            _ => continue,
        };
    }
    if oblique && crit.orth_only() {
        return Err(rc_err(198, format!("{} rotation is orthogonal only", crit.name())));
    }
    Ok(RotOpts { crit, oblique, normalize: cmd.has_opt("normalize", 4) })
}

pub struct RotResult {
    /// rotated loadings (p x k)
    pub l: Matrix,
    /// rotation matrix: L = A T (orthogonal), L = A inv(T') (oblique)
    pub t: Matrix,
    /// correlation of rotated factors: T'T
    pub phi: Matrix,
    pub fmin: f64,
    pub class: String,
    pub ctitle: String,
}

/// Criterion value and gradient.
fn vgq(crit: &Crit, l: &Matrix) -> (f64, Matrix) {
    let p = l.rows;
    let k = l.cols;
    let l2: Vec<f64> = l.data.iter().map(|x| x * x).collect();
    let mk = |data: Vec<f64>| Matrix { rows: p, cols: k, data, rownames: vec![], colnames: vec![] };
    // L2 (row-block) times (11' - I) over columns: for each row, sum of other columns
    let other_cols = |m: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; p * k];
        for i in 0..p {
            let s: f64 = (0..k).map(|j| m[i * k + j]).sum();
            for j in 0..k {
                out[i * k + j] = s - m[i * k + j];
            }
        }
        out
    };
    let other_rows = |m: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; p * k];
        for j in 0..k {
            let s: f64 = (0..p).map(|i| m[i * k + j]).sum();
            for i in 0..p {
                out[i * k + j] = s - m[i * k + j];
            }
        }
        out
    };
    match crit {
        Crit::Quartimax => {
            let f = -l2.iter().map(|x| x * x).sum::<f64>() / 4.0;
            (f, mk(l.data.iter().map(|x| -x * x * x).collect()))
        }
        Crit::Vgpf | Crit::Varimax => {
            // -1/4 sum of column variances of L^2 (times p)
            let mut ql = l2.clone();
            for j in 0..k {
                let m: f64 = (0..p).map(|i| l2[i * k + j]).sum::<f64>() / p as f64;
                for i in 0..p {
                    ql[i * k + j] -= m;
                }
            }
            let f = -ql.iter().map(|x| x * x).sum::<f64>() / 4.0;
            (f, mk(l.data.iter().zip(&ql).map(|(a, q)| -a * q).collect()))
        }
        Crit::Oblimin(_) | Crit::Quartimin => {
            let gam = if let Crit::Oblimin(g) = crit { *g } else { 0.0 };
            let mut x = other_cols(&l2);
            if gam != 0.0 {
                // (I - gam/p 11') X
                for j in 0..k {
                    let m: f64 = (0..p).map(|i| x[i * k + j]).sum::<f64>() * gam / p as f64;
                    for i in 0..p {
                        x[i * k + j] -= m;
                    }
                }
            }
            let f = l2.iter().zip(&x).map(|(a, b)| a * b).sum::<f64>() / 4.0;
            (f, mk(l.data.iter().zip(&x).map(|(a, b)| a * b).collect()))
        }
        Crit::Equamax | Crit::Parsimax | Crit::Cf(_) => {
            let kap = match crit {
                Crit::Equamax => k as f64 / (2.0 * p as f64),
                Crit::Parsimax => (k as f64 - 1.0) / (p as f64 + k as f64 - 2.0),
                Crit::Cf(x) => *x,
                _ => 0.0,
            };
            let xn = other_cols(&l2);
            let xm = other_rows(&l2);
            let f1 = (1.0 - kap) * l2.iter().zip(&xn).map(|(a, b)| a * b).sum::<f64>() / 4.0;
            let f2 = kap * l2.iter().zip(&xm).map(|(a, b)| a * b).sum::<f64>() / 4.0;
            let gq: Vec<f64> = (0..p * k).map(|t| (1.0 - kap) * l.data[t] * xn[t] + kap * l.data[t] * xm[t]).collect();
            (f1 + f2, mk(gq))
        }
        Crit::Entropy => {
            let lg: Vec<f64> = l2.iter().map(|x| if *x > 0.0 { x.ln() } else { 0.0 }).collect();
            let f = -l2.iter().zip(&lg).map(|(a, b)| a * b).sum::<f64>() / 2.0;
            (f, mk((0..p * k).map(|t| -(l.data[t] * lg[t] + l.data[t])).collect()))
        }
        Crit::Oblimax => {
            let s4: f64 = l2.iter().map(|x| x * x).sum();
            let s2: f64 = l2.iter().sum();
            let f = -(s4.ln() - 2.0 * s2.ln());
            (f, mk(l.data.iter().map(|x| -(4.0 * x * x * x / s4 - 4.0 * x / s2)).collect()))
        }
        Crit::Bentler => {
            let l2m = mk(l2.clone());
            let m = l2m.transpose().mul(&l2m).unwrap();
            let d: Vec<f64> = m.diag();
            let f = -(mvutil::ln_det(&m) - d.iter().map(|x| x.ln()).sum::<f64>()) / 4.0;
            let mi = mvutil::sym_inv(&m);
            let mut dd = mi.clone();
            for j in 0..k {
                dd[(j, j)] -= 1.0 / d[j];
            }
            let x = l2m.mul(&dd).unwrap();
            (f, mk((0..p * k).map(|t| -l.data[t] * x.data[t]).collect()))
        }
        Crit::Tandem1 | Crit::Tandem2 => {
            let ll = l.mul(&l.transpose()).unwrap();
            let ll2: Vec<f64> = ll.data.iter().map(|x| x * x).collect();
            let l2m = mk(l2.clone());
            let ll2m = Matrix { rows: p, cols: p, data: ll2.clone(), rownames: vec![], colnames: vec![] };
            let one_minus = Matrix { rows: p, cols: p, data: ll2.iter().map(|x| 1.0 - x).collect(), rownames: vec![], colnames: vec![] };
            let l2l2t = l2m.mul(&l2m.transpose()).unwrap();
            let had = Matrix { rows: p, cols: p, data: ll.data.iter().zip(&l2l2t.data).map(|(a, b)| a * b).collect(), rownames: vec![], colnames: vec![] };
            let gq2 = had.mul(l).unwrap().scale(4.0);
            if matches!(crit, Crit::Tandem1) {
                let a = ll2m.mul(&l2m).unwrap();
                let f = -l2.iter().zip(&a.data).map(|(x, y)| x * y).sum::<f64>();
                let gq1: Vec<f64> = (0..p * k).map(|t| 4.0 * l.data[t] * a.data[t]).collect();
                (f, mk((0..p * k).map(|t| -gq1[t] - gq2.data[t]).collect()))
            } else {
                let a = one_minus.mul(&l2m).unwrap();
                let f = l2.iter().zip(&a.data).map(|(x, y)| x * y).sum::<f64>();
                let gq1: Vec<f64> = (0..p * k).map(|t| 4.0 * l.data[t] * a.data[t]).collect();
                (f, mk((0..p * k).map(|t| gq1[t] - gq2.data[t]).collect()))
            }
        }
        Crit::Target(h) => {
            let d: Vec<f64> = (0..p * k).map(|t| l.data[t] - h.data.get(t).copied().unwrap_or(0.0)).collect();
            (d.iter().map(|x| x * x).sum::<f64>() / 2.0, mk(d))
        }
        Crit::Partial(h, w) => {
            let d: Vec<f64> = (0..p * k).map(|t| w.data.get(t).copied().unwrap_or(0.0) * (l.data[t] - h.data.get(t).copied().unwrap_or(0.0))).collect();
            let wd: Vec<f64> = (0..p * k).map(|t| 2.0 * w.data.get(t).copied().unwrap_or(0.0) * d[t]).collect();
            (d.iter().map(|x| x * x).sum::<f64>(), mk(wd))
        }
        _ => (0.0, mk(vec![0.0; p * k])),
    }
}

fn frob(m: &Matrix) -> f64 {
    m.data.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Gradient projection, orthogonal.
fn gpf_orth(a: &Matrix, crit: &Crit, t0: Matrix) -> (Matrix, f64) {
    let mut t = t0;
    let mut al = 1.0;
    let mut l = a.mul(&t).unwrap();
    let (mut f, gq) = vgq(crit, &l);
    let mut g = a.transpose().mul(&gq).unwrap();
    for _ in 0..2000 {
        let m = t.transpose().mul(&g).unwrap();
        let s = m.add(&m.transpose()).unwrap().scale(0.5);
        let gp = g.sub(&t.mul(&s).unwrap()).unwrap();
        let sn = frob(&gp);
        if sn < 1e-10 {
            break;
        }
        al *= 2.0;
        let mut accepted = None;
        for _ in 0..=10 {
            let x = t.sub(&gp.scale(al)).unwrap();
            let (tt, _) = polar(&x);
            let lt = a.mul(&tt).unwrap();
            let (ft, gqt) = vgq(crit, &lt);
            if ft < f - 0.5 * sn * sn * al {
                accepted = Some((tt, lt, ft, gqt));
                break;
            }
            al /= 2.0;
        }
        match accepted {
            Some((tt, lt, ft, gqt)) => {
                t = tt;
                l = lt;
                f = ft;
                g = a.transpose().mul(&gqt).unwrap();
            }
            None => break,
        }
    }
    let _ = l;
    (t, f)
}

/// Gradient projection, oblique (T with unit-length columns; L = A inv(T)').
fn gpf_obl(a: &Matrix, crit: &Crit, t0: Matrix) -> (Matrix, f64) {
    let k = a.cols;
    let mut t = t0;
    let mut al = 1.0;
    let ti = t.inverse().unwrap_or_else(|_| Matrix::identity(k));
    let mut l = a.mul(&ti.transpose()).unwrap();
    let (mut f, gq) = vgq(crit, &l);
    let mut g = l.transpose().mul(&gq).unwrap().mul(&ti).unwrap().transpose().scale(-1.0);
    for _ in 0..5000 {
        // Gp = G - T diag(colsums(T * G))
        let mut gp = g.clone();
        for j in 0..k {
            let s: f64 = (0..k).map(|i| t[(i, j)] * g[(i, j)]).sum();
            for i in 0..k {
                gp[(i, j)] -= t[(i, j)] * s;
            }
        }
        let sn = frob(&gp);
        if sn < 1e-10 {
            break;
        }
        al *= 2.0;
        let mut accepted = None;
        for _ in 0..=10 {
            let mut x = t.sub(&gp.scale(al)).unwrap();
            for j in 0..k {
                let nrm: f64 = (0..k).map(|i| x[(i, j)] * x[(i, j)]).sum::<f64>().sqrt();
                for i in 0..k {
                    x[(i, j)] /= nrm;
                }
            }
            let Ok(xi) = x.inverse() else {
                al /= 2.0;
                continue;
            };
            let lt = a.mul(&xi.transpose()).unwrap();
            let (ft, gqt) = vgq(crit, &lt);
            if f - ft > 0.5 * sn * sn * al {
                accepted = Some((x, xi, lt, ft, gqt));
                break;
            }
            al /= 2.0;
        }
        match accepted {
            Some((tt, tti, lt, ft, gqt)) => {
                t = tt;
                l = lt;
                f = ft;
                g = l.transpose().mul(&gqt).unwrap().mul(&tti).unwrap().transpose().scale(-1.0);
            }
            None => break,
        }
    }
    (t, f)
}

/// Kaiser's varimax (raw); returns the orthogonal rotation T.
fn varimax_classic(a: &Matrix) -> Matrix {
    let p = a.rows;
    let k = a.cols;
    let mut t = Matrix::identity(k);
    let mut d = 0.0;
    for _ in 0..10000 {
        let z = a.mul(&t).unwrap();
        let mut b = Matrix::zeros(p, k);
        for j in 0..k {
            let s2: f64 = (0..p).map(|i| z[(i, j)] * z[(i, j)]).sum::<f64>() / p as f64;
            for i in 0..p {
                b[(i, j)] = z[(i, j)].powi(3) - z[(i, j)] * s2;
            }
        }
        let bb = a.transpose().mul(&b).unwrap();
        let (tn, sv) = polar(&bb);
        t = tn;
        let dpast = d;
        d = sv;
        if d < dpast * (1.0 + 1e-12) {
            break;
        }
    }
    t
}

/// Rotate the first `nf` columns of `a` (the rest are left as they are).
pub fn rotate(a: &Matrix, o: &RotOpts, nf: usize) -> Result<RotResult, String> {
    let p = a.rows;
    let kall = a.cols;
    let k = nf.min(kall).max(1);
    let mut sub = mvutil::left_cols(a, k);
    let sub0 = sub.clone();
    // Kaiser normalization over the rotated columns
    let h: Vec<f64> = (0..p).map(|i| (0..k).map(|j| sub[(i, j)] * sub[(i, j)]).sum::<f64>().sqrt()).collect();
    if o.normalize {
        for i in 0..p {
            for j in 0..k {
                if h[i] > 0.0 {
                    sub[(i, j)] /= h[i];
                }
            }
        }
    }
    let (tk, fmin, oblique) = match &o.crit {
        Crit::Varimax if !o.oblique => {
            let t = varimax_classic(&sub);
            let (f, _) = vgq(&Crit::Vgpf, &sub.mul(&t).unwrap());
            (t, f, false)
        }
        Crit::Promax(pow) => {
            let tv = varimax_classic(&sub);
            let lv = sub.mul(&tv).unwrap();
            // target: signed power
            let mut q = lv.clone();
            for x in q.data.iter_mut() {
                *x = x.signum() * x.abs().powf(*pow);
            }
            // oblique Procrustes of the (unnormalized) loadings toward the target:
            // U = (A'A)^-1 A'Q, columns scaled so that diag(inv(U'U)) = 1
            let u0 = mvutil::sym_inv(&sub0.transpose().mul(&sub0).unwrap()).mul(&sub0.transpose().mul(&q).unwrap()).unwrap();
            let ui = u0.transpose().mul(&u0).unwrap().inverse()?;
            let mut u = u0.clone();
            for j in 0..k {
                let s = ui[(j, j)].sqrt();
                for i in 0..k {
                    u[(i, j)] *= s;
                }
            }
            // L = A U = A inv(T')  =>  T = inv(U)'
            let t = u.inverse()?.transpose();
            (t, 0.0, true)
        }
        c => {
            if o.oblique {
                let (t, f) = gpf_obl(&sub, c, Matrix::identity(k));
                (t, f, true)
            } else {
                let (t, f) = gpf_orth(&sub, c, Matrix::identity(k));
                (t, f, false)
            }
        }
    };
    let is_promax = matches!(o.crit, Crit::Promax(_));
    let base = if is_promax { &sub0 } else { &sub };
    let mut lk = if oblique { base.mul(&tk.inverse()?.transpose()).unwrap() } else { base.mul(&tk).unwrap() };
    if o.normalize && !is_promax {
        for i in 0..p {
            for j in 0..k {
                lk[(i, j)] *= h[i];
            }
        }
    }
    // full matrices with the unrotated trailing columns
    let mut l = a.clone();
    let mut t = Matrix::identity(kall);
    for i in 0..p {
        for j in 0..k {
            l[(i, j)] = lk[(i, j)];
        }
    }
    for i in 0..k {
        for j in 0..k {
            t[(i, j)] = tk[(i, j)];
        }
    }
    // orient columns to positive sums
    for j in 0..k {
        let s: f64 = (0..p).map(|i| l[(i, j)]).sum();
        if s < 0.0 {
            for i in 0..p {
                l[(i, j)] = -l[(i, j)];
            }
            for i in 0..kall {
                t[(i, j)] = -t[(i, j)];
            }
        }
    }
    let phi = t.transpose().mul(&t).unwrap();
    let class = if oblique { "oblique" } else { "orthogonal" };
    Ok(RotResult { l, t, phi, fmin, class: class.into(), ctitle: o.crit.ctitle() })
}

/// Reorder the first `k` columns by `order` (indices into the old columns).
pub fn permute(r: &mut RotResult, order: &[usize]) {
    let (p, kall) = (r.l.rows, r.l.cols);
    let (l0, t0) = (r.l.clone(), r.t.clone());
    for (nj, &oj) in order.iter().enumerate() {
        for i in 0..p {
            r.l[(i, nj)] = l0[(i, oj)];
        }
        for i in 0..kall {
            r.t[(i, nj)] = t0[(i, oj)];
        }
    }
    r.phi = r.t.transpose().mul(&r.t).unwrap();
}

/// Text such as `orthogonal varimax (Kaiser off)`.
pub fn rotation_text(e: &Results) -> String {
    format!(
        "{} {} (Kaiser {})",
        e.macros.get("r_class").cloned().unwrap_or_default(),
        e.macros.get("r_criterion").cloned().unwrap_or_default(),
        if e.macros.get("r_normalization").map(String::as_str) == Some("kaiser") { "on" } else { "off" }
    )
}

// ---------------------------------------------------------------------------
// rotatemat
// ---------------------------------------------------------------------------

pub struct Rotatemat;

impl CommandHandler for Rotatemat {
    fn name(&self) -> &'static str {
        "rotatemat"
    }

    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let name = cmd.main.split_whitespace().next().ok_or_else(|| rc_err(100, "matrix name required"))?;
        let a = st.matrices.get(name).cloned().ok_or_else(|| rc_err(111, format!("matrix {name} not found")))?;
        let o = parse_opts(st, cmd)?;
        let nf = mvutil::opt_usize(cmd, "factors", 1)?.or(mvutil::opt_usize(cmd, "components", 4)?).unwrap_or(a.cols);
        let r = rotate(&a, &o, nf)?;
        let rn = if a.rownames.len() == a.rows { a.rownames.clone() } else { mvutil::names("r", a.rows) };
        let cn = if a.colnames.len() == a.cols { a.colnames.clone() } else { mvutil::names("c", a.cols) };
        let lm = r.l.clone().with_names(rn, cn.clone());
        let tm = r.t.clone().with_names(cn.clone(), cn.clone());
        let fmt_ = cmd.opt("format", 3).unwrap_or("%9.5f").to_string();
        let mut out = vec![String::new()];
        let rtext = format!("{} {}{}", r.class, o.crit.name(), if o.normalize { ", Kaiser normalization" } else { "" });
        if !cmd.has_opt("noloading", 5) {
            out.push(format!("Rotation of {name}[{},{}]", a.rows, a.cols));
            out.push(String::new());
            out.push(format!("    Criterion       {}", o.crit.name()));
            out.push(format!("    Rotation class  {}", r.class));
            out.push(format!("    Normalization   {}", if o.normalize { "Kaiser" } else { "none" }));
            out.push(String::new());
            out.push("Rotated factors".into());
            out.push(String::new());
            out.extend(mvutil::matlist(&lm, "", &fmt_, 9, false, 4));
        }
        if !cmd.has_opt("norotation", 5) {
            out.push(String::new());
            out.push("Rotation matrix".into());
            out.push(String::new());
            out.extend(mvutil::matlist(&tm, "", &fmt_, 9, false, 4));
        }
        let _ = rtext;
        let res = st.begin_r();
        res.set("f", r.fmin);
        res.set_str("criterion", o.crit.name());
        res.set_str("class", r.class.clone());
        res.set_str("ctitle", o.crit.ctitle());
        res.set_str("normalization", if o.normalize { "kaiser" } else { "none" });
        res.set_mat("AT", lm);
        res.set_mat("T", tm);
        if r.class == "oblique" {
            res.set_mat("Phi", r.phi.clone().with_names(cn.clone(), cn));
        }
        let _ = util::level;
        Ok(out.join("\n"))
    }
}
