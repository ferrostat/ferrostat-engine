//! Spatial autoregressive models: `spregress` (GS2SLS and maximum
//! likelihood), `spivregress`, `spxtregress` (fixed and random effects) and
//! their postestimation (`estat impact`, `predict`), plus `estat moran`
//! after `regress`.
//!
//! Model: y = λ W y + X β + (W X) θ + u,  u = ρ M u + ε.
//!
//! * ML: the log likelihood uses ln|I − λW| = Σ ln|1 − λω_i| over the
//!   eigenvalues ω of W; it is maximized over (λ, ρ) with β and σ²
//!   concentrated out, then polished by Newton steps on the full likelihood;
//!   the variance is the inverse of the observed information.
//! * GS2SLS: Kelejian–Prucha generalized spatial two-stage least squares with
//!   instruments (X, WX, W²X, and M times those when there is an error lag),
//!   GMM estimation of ρ from the two quadratic moment conditions (Drukker,
//!   Egger and Prucha for homoskedastic errors; Arraiz, Drukker, Kelejian
//!   and Prucha for `heteroskedastic`) and their joint asymptotic variance.

use std::collections::HashMap;

use ferrostat_core::missing::SYSMISS;
use ferrostat_core::{DataSet, VarType, Variable};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::{rc_err, Env};
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::sp_data::{align, dlogdet, eigenvalues, get_spmat, logdet, need_spset, panel_vars, param_bounds};
use crate::store::Store;
use crate::{est, util};

// ---------------------------------------------------------------------------
// Sparse rows (weighting matrices are usually sparse)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Sparse {
    pub n: usize,
    pub rows: Vec<Vec<(usize, f64)>>,
}

impl Sparse {
    pub fn from_dense(m: &Matrix) -> Sparse {
        let rows = (0..m.rows).map(|i| m.row(i).iter().enumerate().filter(|(_, v)| **v != 0.0).map(|(j, v)| (j, *v)).collect()).collect();
        Sparse { n: m.rows, rows }
    }
    pub fn mv(&self, x: &[f64]) -> Vec<f64> {
        self.rows.iter().map(|r| r.iter().map(|&(j, v)| v * x[j]).sum()).collect()
    }
    /// Apply blockwise to stacked periods of length n.
    pub fn mv_blocks(&self, x: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(x.len());
        for b in x.chunks(self.n) {
            out.extend(self.mv(b));
        }
        out
    }
    pub fn t(&self) -> Sparse {
        let mut rows = vec![Vec::new(); self.n];
        for (i, r) in self.rows.iter().enumerate() {
            for &(j, v) in r {
                rows[j].push((i, v));
            }
        }
        Sparse { n: self.n, rows }
    }
    pub fn mul(&self, o: &Sparse) -> Sparse {
        let mut acc = vec![0.0; self.n];
        let mut touched: Vec<usize> = Vec::new();
        let mut rows = Vec::with_capacity(self.n);
        for r in &self.rows {
            for &(k, a) in r {
                for &(j, b) in &o.rows[k] {
                    if acc[j] == 0.0 {
                        touched.push(j);
                    }
                    acc[j] += a * b;
                }
            }
            touched.sort_unstable();
            touched.dedup();
            let row: Vec<(usize, f64)> = touched.iter().map(|&j| (j, acc[j])).filter(|x| x.1 != 0.0).collect();
            for &j in &touched {
                acc[j] = 0.0;
            }
            touched.clear();
            rows.push(row);
        }
        Sparse { n: self.n, rows }
    }
    /// a*self + b*o
    pub fn lin(&self, a: f64, o: &Sparse, b: f64) -> Sparse {
        let rows = self
            .rows
            .iter()
            .zip(&o.rows)
            .map(|(r1, r2)| {
                let mut m: std::collections::BTreeMap<usize, f64> = Default::default();
                for &(j, v) in r1 {
                    *m.entry(j).or_default() += a * v;
                }
                for &(j, v) in r2 {
                    *m.entry(j).or_default() += b * v;
                }
                m.into_iter().filter(|x| x.1 != 0.0).collect()
            })
            .collect();
        Sparse { n: self.n, rows }
    }
    pub fn diag(&self) -> Vec<f64> {
        self.rows.iter().enumerate().map(|(i, r)| r.iter().find(|x| x.0 == i).map(|x| x.1).unwrap_or(0.0)).collect()
    }
    pub fn identity(n: usize) -> Sparse {
        Sparse { n, rows: (0..n).map(|i| vec![(i, 1.0)]).collect() }
    }
    pub fn from_diag(d: &[f64]) -> Sparse {
        Sparse { n: d.len(), rows: d.iter().enumerate().map(|(i, v)| if *v != 0.0 { vec![(i, *v)] } else { vec![] }).collect() }
    }
    pub fn quad(&self, x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(self.mv(y)).map(|(a, b)| a * b).sum()
    }
    pub fn trace(&self) -> f64 {
        self.diag().iter().sum()
    }
}

/// Σ_ij A_ij B_ij s_i s_j for sparse A, B (rows sorted by column).
fn weighted_frob(a: &Sparse, b: &Sparse, s: Option<&[f64]>) -> f64 {
    let mut tot = 0.0;
    for i in 0..a.n {
        let (ra, rb) = (&a.rows[i], &b.rows[i]);
        let (mut p, mut q) = (0, 0);
        while p < ra.len() && q < rb.len() {
            match ra[p].0.cmp(&rb[q].0) {
                std::cmp::Ordering::Less => p += 1,
                std::cmp::Ordering::Greater => q += 1,
                std::cmp::Ordering::Equal => {
                    let j = ra[p].0;
                    let w = s.map(|s| s[i] * s[j]).unwrap_or(1.0);
                    tot += ra[p].1 * rb[q].1 * w;
                    p += 1;
                    q += 1;
                }
            }
        }
    }
    tot
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// A'B for column lists.
fn cross(a: &[Vec<f64>], b: &[Vec<f64>]) -> Matrix {
    let mut m = Matrix::zeros(a.len(), b.len());
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            m[(i, j)] = dot(x, y);
        }
    }
    m
}

fn mat_vec(m: &Matrix, v: &[f64]) -> Vec<f64> {
    (0..m.rows).map(|i| m.row(i).iter().zip(v).map(|(a, b)| a * b).sum()).collect()
}

fn mmul(a: &Matrix, b: &Matrix) -> Matrix {
    a.mul(b).expect("conformable")
}

fn inv(m: &Matrix) -> Matrix {
    m.sym_inverse(1e-13).0
}

/// Linear combination Σ_j cols_j * b_j.
fn comb(cols: &[Vec<f64>], b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for (c, bj) in cols.iter().zip(b) {
        for i in 0..n {
            out[i] += c[i] * bj;
        }
    }
    out
}

/// Keep linearly independent columns (modified Gram–Schmidt).
fn independent(cols: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut keep = Vec::new();
    for c in cols {
        let norm0 = dot(&c, &c).sqrt();
        if norm0 == 0.0 {
            continue;
        }
        let mut r = c.clone();
        for q in &basis {
            let p = dot(&r, q);
            for (ri, qi) in r.iter_mut().zip(q) {
                *ri -= p * qi;
            }
        }
        let nr = dot(&r, &r).sqrt();
        if nr > 1e-9 * norm0 {
            basis.push(r.iter().map(|v| v / nr).collect());
            keep.push(c);
        }
    }
    keep
}

/// Solve (I − λW) x = b by fixed-point iteration, falling back to a dense solve.
pub fn solve_ilw(w: &Sparse, lam: f64, b: &[f64]) -> Vec<f64> {
    let mut x = b.to_vec();
    for _ in 0..2000 {
        let wx = w.mv(&x);
        let nx: Vec<f64> = b.iter().zip(&wx).map(|(bi, v)| bi + lam * v).collect();
        let diff = nx.iter().zip(&x).map(|(a, c)| (a - c).abs()).fold(0.0, f64::max);
        let scale = nx.iter().map(|v| v.abs()).fold(1e-300, f64::max);
        x = nx;
        if diff <= 1e-14 * scale {
            return x;
        }
    }
    let n = w.n;
    let mut a = Matrix::identity(n);
    for (i, r) in w.rows.iter().enumerate() {
        for &(j, v) in r {
            a[(i, j)] -= lam * v;
        }
    }
    match a.inverse() {
        Ok(ai) => mat_vec(&ai, b),
        Err(_) => x,
    }
}

/// (I − λW)^{-1} as a dense matrix.
fn ilw_inverse(w: &Sparse, lam: f64) -> Result<Matrix, String> {
    let n = w.n;
    let mut a = Matrix::identity(n);
    for (i, r) in w.rows.iter().enumerate() {
        for &(j, v) in r {
            a[(i, j)] -= lam * v;
        }
    }
    a.inverse().map_err(|_| rc_err(430, "I - lambda*W is singular"))
}

// ---------------------------------------------------------------------------
// Specification and data preparation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct Spec {
    depvar: String,
    exog: String,
    endog: Vec<String>,
    inst: Vec<String>,
    dl: Option<String>,
    el: Option<String>,
    iv: Vec<(String, Vec<String>)>,
}

fn parse_spec(ds: &DataSet, cmd: &Command, iv_syntax: bool) -> Result<Spec, String> {
    let mut main = cmd.main.trim().to_string();
    let mut sp = Spec::default();
    if iv_syntax {
        if let (Some(a), Some(b)) = (main.find('('), main.rfind(')')) {
            let inner = main[a + 1..b].to_string();
            let (en, ins) = inner.split_once('=').ok_or_else(|| rc_err(198, "syntax is (varlist = varlist)"))?;
            sp.endog = crate::varlist::expand(ds, en)?;
            sp.inst = crate::varlist::expand(ds, ins)?;
            main = format!("{} {}", &main[..a], &main[b + 1..]);
        }
    }
    let mut it = main.split_whitespace();
    sp.depvar = crate::varlist::resolve_name(ds, it.next().ok_or_else(|| rc_err(100, "varlist required"))?)?;
    sp.exog = it.collect::<Vec<_>>().join(" ");
    for (n, a) in &cmd.opts {
        let n = n.to_lowercase();
        let a = a.clone().unwrap_or_default();
        if n.starts_with("dvar") {
            if sp.dl.is_some() {
                return Err(rc_err(198, "only one dvarlag() matrix is supported"));
            }
            sp.dl = Some(a.trim().to_string());
        } else if n.starts_with("errorl") {
            if sp.el.is_some() {
                return Err(rc_err(198, "only one errorlag() matrix is supported"));
            }
            sp.el = Some(a.trim().to_string());
        } else if n.starts_with("ivar") {
            let (w, vars) = a.split_once(':').ok_or_else(|| rc_err(198, "ivarlag(W: varlist) required"))?;
            sp.iv.push((w.trim().to_string(), crate::varlist::expand(ds, vars)?));
        }
    }
    Ok(sp)
}

#[allow(dead_code)]
struct Prep {
    /// units per period, periods
    n: usize,
    t: usize,
    /// data rows in estimation order (period-major, units sorted by ID)
    rows: Vec<usize>,
    ids: Vec<f64>,
    y: Vec<f64>,
    xnames: Vec<String>,
    xcols: Vec<Vec<f64>>,
    has_cons: bool,
    /// (matrix, variable, lagged column)
    lags: Vec<(String, String, Vec<f64>)>,
    endog: Vec<(String, Vec<f64>)>,
    inst: Vec<Vec<f64>>,
    mats: HashMap<String, Sparse>,
    w_places: HashMap<String, usize>,
    mask: Vec<bool>,
    nobs_all: usize,
    panel_var: Option<String>,
}

fn prepare(ds: &DataSet, st: &Store, cmd: &Command, sp: &Spec, panel: bool) -> Result<Prep, String> {
    let s = need_spset(ds)?;
    let pv = panel_vars(st);
    if panel && pv.is_none() {
        return Err(rc_err(459, "data not xtset"));
    }
    let mut extra = vec![s.id.clone()];
    extra.extend(sp.endog.iter().cloned());
    extra.extend(sp.inst.iter().cloned());
    for (_, v) in &sp.iv {
        extra.extend(v.iter().cloned());
    }
    if let Some((p, t)) = &pv {
        if panel {
            extra.push(p.clone());
            extra.push(t.clone());
        }
    }
    if cmd.weight.is_some() {
        return Err(rc_err(101, "weights not allowed"));
    }
    let noconstant = cmd.has_opt("noconstant", 3);
    let mut d = est::build(ds, st, cmd, est::DesignSpec { depvar: Some(&sp.depvar), indep: &sp.exog, noconstant, weights: "", extra })?;
    est::drop_collinear(&mut d);
    let idv = util::col(ds, &s.id)?;
    let timev = if panel { Some(util::col(ds, &pv.as_ref().unwrap().1)?) } else { None };
    // order: by time then ID
    let mut order: Vec<usize> = (0..d.n()).collect();
    order.sort_by(|&a, &b| {
        let (ra, rb) = (d.rows[a], d.rows[b]);
        let ta = timev.as_ref().map(|t| t[ra]).unwrap_or(0.0);
        let tb = timev.as_ref().map(|t| t[rb]).unwrap_or(0.0);
        ta.total_cmp(&tb).then(idv[ra].total_cmp(&idv[rb]))
    });
    let rows: Vec<usize> = order.iter().map(|&k| d.rows[k]).collect();
    let nt = rows.len();
    if nt == 0 {
        return Err(rc_err(2000, "no observations"));
    }
    // periods
    let (n, t) = if let Some(tv) = &timev {
        let mut periods: Vec<f64> = rows.iter().map(|&r| tv[r]).collect();
        periods.dedup();
        let t = periods.len();
        if nt % t != 0 {
            return Err(rc_err(459, "panel is not strongly balanced in the estimation sample\n    see spbalance"));
        }
        let n = nt / t;
        for p in 1..t {
            for i in 0..n {
                if idv[rows[p * n + i]] != idv[rows[i]] {
                    return Err(rc_err(459, "panel is not strongly balanced in the estimation sample\n    see spbalance"));
                }
            }
        }
        (n, t)
    } else {
        for k in 1..nt {
            if idv[rows[k]] == idv[rows[k - 1]] {
                return Err(rc_err(459, format!("{} does not uniquely identify the observations", s.id)));
            }
        }
        (nt, 1)
    };
    let ids: Vec<f64> = rows[..n].iter().map(|&r| idv[r]).collect();
    let force = cmd.has_opt("force", 5);
    let mut names: Vec<String> = Vec::new();
    names.extend(sp.dl.iter().cloned());
    names.extend(sp.el.iter().cloned());
    names.extend(sp.iv.iter().map(|x| x.0.clone()));
    let mut mats = HashMap::new();
    let mut w_places = HashMap::new();
    for nm in names {
        if mats.contains_key(&nm) {
            continue;
        }
        let m = get_spmat(st, &nm)?;
        w_places.insert(nm.clone(), m.ids.len());
        mats.insert(nm.clone(), Sparse::from_dense(&align(&m, &ids, &nm, force)?));
    }
    let act = d.active();
    let xnames: Vec<String> = act.iter().map(|&j| d.names[j].clone()).collect();
    let xcols: Vec<Vec<f64>> = act.iter().map(|&j| order.iter().map(|&k| d.cols[j][k]).collect()).collect();
    let y: Vec<f64> = order.iter().map(|&k| d.y[k]).collect();
    let mut lags = Vec::new();
    for (w, vars) in &sp.iv {
        for v in vars {
            let x = util::col(ds, v)?;
            let xs: Vec<f64> = rows.iter().map(|&r| x[r]).collect();
            lags.push((w.clone(), v.clone(), mats[w].mv_blocks(&xs)));
        }
    }
    let endog = sp.endog.iter().map(|v| util::col(ds, v).map(|x| (v.clone(), rows.iter().map(|&r| x[r]).collect()))).collect::<Result<Vec<_>, _>>()?;
    let inst = sp.inst.iter().map(|v| util::col(ds, v).map(|x| rows.iter().map(|&r| x[r]).collect())).collect::<Result<Vec<_>, _>>()?;
    Ok(Prep { n, t, rows, ids, y, has_cons: xnames.iter().any(|x| x == "_cons"), xnames, xcols, lags, endog, inst, mats, w_places, mask: d.mask.clone(), nobs_all: ds.n_obs, panel_var: pv.map(|p| p.0) })
}

/// Display order and names of the coefficients.
struct Layout {
    names: Vec<String>,
    /// internal parameter index for each displayed coefficient
    idx: Vec<usize>,
}

/// Internal parameter vector: [β (x), θ (lags), endog coefs, λ?, ρ?, extra...].
fn layout(p: &Prep, sp: &Spec, extra: &[String]) -> Layout {
    let y = &sp.depvar;
    let kx = p.xcols.len();
    let kl = p.lags.len();
    let ke = p.endog.len();
    let mut names = Vec::new();
    let mut idx = Vec::new();
    // depvar equation: endogenous first, then exogenous in order
    for (k, (n, _)) in p.endog.iter().enumerate() {
        names.push(format!("{y}:{n}"));
        idx.push(kx + kl + k);
    }
    for (k, n) in p.xnames.iter().enumerate() {
        if n == "_cons" {
            continue;
        }
        names.push(format!("{y}:{n}"));
        idx.push(k);
    }
    if let Some(k) = p.xnames.iter().position(|n| n == "_cons") {
        names.push(format!("{y}:_cons"));
        idx.push(k);
    }
    let base = kx + kl + ke;
    let lam_i = sp.dl.as_ref().map(|_| base);
    let rho_i = sp.el.as_ref().map(|_| base + lam_i.is_some() as usize);
    let mut mats: Vec<String> = Vec::new();
    for m in sp.dl.iter().chain(p.lags.iter().map(|l| &l.0)).chain(sp.el.iter()) {
        if !mats.contains(m) {
            mats.push(m.clone());
        }
    }
    for m in &mats {
        if sp.dl.as_deref() == Some(m.as_str()) {
            names.push(format!("{m}:{y}"));
            idx.push(lam_i.unwrap());
        }
        for (k, l) in p.lags.iter().enumerate() {
            if &l.0 == m {
                names.push(format!("{m}:{}", l.1));
                idx.push(kx + k);
            }
        }
        if sp.el.as_deref() == Some(m.as_str()) {
            names.push(format!("{m}:e.{y}"));
            idx.push(rho_i.unwrap());
        }
    }
    let nparam = base + lam_i.is_some() as usize + rho_i.is_some() as usize;
    for (k, e) in extra.iter().enumerate() {
        names.push(e.clone());
        idx.push(nparam + k);
    }
    Layout { names, idx }
}

fn reorder(l: &Layout, b: &[f64], v: &Matrix) -> (Vec<f64>, Matrix) {
    let k = l.idx.len();
    let bb = l.idx.iter().map(|&i| b[i]).collect();
    let mut vv = Matrix::zeros(k, k);
    for a in 0..k {
        for c in 0..k {
            vv[(a, c)] = v[(l.idx[a], l.idx[c])];
        }
    }
    (bb, vv)
}

/// Wald test of the coefficients whose names satisfy `pick`.
fn wald(names: &[String], b: &[f64], v: &Matrix, pick: &dyn Fn(&str) -> bool) -> (f64, usize) {
    let idx: Vec<usize> = (0..names.len()).filter(|&j| pick(&names[j])).collect();
    let k = idx.len();
    if k == 0 {
        return (0.0, 0);
    }
    let mut vs = Matrix::zeros(k, k);
    for (a, &i) in idx.iter().enumerate() {
        for (c, &j) in idx.iter().enumerate() {
            vs[(a, c)] = v[(i, j)];
        }
    }
    let vi = inv(&vs);
    let bs: Vec<f64> = idx.iter().map(|&i| b[i]).collect();
    (dot(&bs, &mat_vec(&vi, &bs)), k)
}

fn squared_corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma).powi(2);
        sbb += (y - mb).powi(2);
    }
    sab * sab / (saa * sbb)
}

// ---------------------------------------------------------------------------
// Maximum likelihood (cross-section and fixed-effects panels)
// ---------------------------------------------------------------------------

struct Ml<'a> {
    y: Vec<f64>,
    wy: Vec<f64>,
    my: Vec<f64>,
    mwy: Vec<f64>,
    x: Vec<Vec<f64>>,
    mx: Vec<Vec<f64>>,
    ew: Option<&'a [(f64, f64)]>,
    em: Option<&'a [(f64, f64)]>,
    tf: f64,
    neff: f64,
}

impl<'a> Ml<'a> {
    fn kx(&self) -> usize {
        self.x.len()
    }
    /// ε and the untransformed residual r = y − λWy − Xβ.
    fn resid(&self, beta: &[f64], lam: f64, rho: f64) -> (Vec<f64>, Vec<f64>) {
        let nt = self.y.len();
        let mut r = vec![0.0; nt];
        let mut eps = vec![0.0; nt];
        let xb = comb(&self.x, beta, nt);
        let mxb = comb(&self.mx, beta, nt);
        for i in 0..nt {
            r[i] = self.y[i] - lam * self.wy[i] - xb[i];
            let mr = self.my[i] - lam * self.mwy[i] - mxb[i];
            eps[i] = r[i] - rho * mr;
        }
        (eps, r)
    }
    fn logdets(&self, lam: f64, rho: f64) -> f64 {
        self.tf * (self.ew.map_or(0.0, |e| logdet(e, lam)) + self.em.map_or(0.0, |e| logdet(e, rho)))
    }
    /// Concentrated log likelihood, with β and σ².
    fn conc(&self, lam: f64, rho: f64) -> (f64, Vec<f64>, f64) {
        let nt = self.y.len();
        let yt: Vec<f64> = (0..nt).map(|i| self.y[i] - lam * self.wy[i] - rho * (self.my[i] - lam * self.mwy[i])).collect();
        let xt: Vec<Vec<f64>> = self.x.iter().zip(&self.mx).map(|(x, mx)| x.iter().zip(mx).map(|(a, b)| a - rho * b).collect()).collect();
        let xx = cross(&xt, &xt);
        let xy: Vec<f64> = xt.iter().map(|c| dot(c, &yt)).collect();
        let beta = mat_vec(&inv(&xx), &xy);
        let fit = comb(&xt, &beta, nt);
        let ssr: f64 = yt.iter().zip(&fit).map(|(a, b)| (a - b).powi(2)).sum();
        let s2 = ssr / self.neff;
        let ll = -0.5 * self.neff * ((2.0 * std::f64::consts::PI * s2).ln() + 1.0) + self.logdets(lam, rho);
        (ll, beta, s2)
    }
    fn ll(&self, th: &[f64], has_l: bool, has_r: bool) -> f64 {
        let k = self.kx();
        let lam = if has_l { th[k] } else { 0.0 };
        let rho = if has_r { th[k + has_l as usize] } else { 0.0 };
        let s2 = th[th.len() - 1];
        if s2 <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let (eps, _) = self.resid(&th[..k], lam, rho);
        let ee = dot(&eps, &eps);
        -0.5 * self.neff * (2.0 * std::f64::consts::PI * s2).ln() + self.logdets(lam, rho) - ee / (2.0 * s2)
    }
    fn grad(&self, th: &[f64], has_l: bool, has_r: bool) -> Vec<f64> {
        let k = self.kx();
        let nt = self.y.len();
        let lam = if has_l { th[k] } else { 0.0 };
        let rho = if has_r { th[k + has_l as usize] } else { 0.0 };
        let s2 = th[th.len() - 1];
        let (eps, r) = self.resid(&th[..k], lam, rho);
        let mut g = Vec::with_capacity(th.len());
        for j in 0..k {
            let s: f64 = (0..nt).map(|i| eps[i] * (self.x[j][i] - rho * self.mx[j][i])).sum();
            g.push(s / s2);
        }
        if has_l {
            let s: f64 = (0..nt).map(|i| eps[i] * (self.wy[i] - rho * self.mwy[i])).sum();
            g.push(self.tf * dlogdet(self.ew.unwrap(), lam) + s / s2);
        }
        if has_r {
            let mxb = comb(&self.mx, &th[..k], nt);
            let s: f64 = (0..nt).map(|i| eps[i] * (self.my[i] - lam * self.mwy[i] - mxb[i])).sum();
            let _ = &r;
            g.push(self.tf * dlogdet(self.em.unwrap(), rho) + s / s2);
        }
        let ee = dot(&eps, &eps);
        g.push(-0.5 * self.neff / s2 + ee / (2.0 * s2 * s2));
        g
    }
}

/// Hessian by central differences of an analytic gradient.
fn num_hessian(th: &[f64], g: &dyn Fn(&[f64]) -> Vec<f64>) -> Matrix {
    let k = th.len();
    let mut h = Matrix::zeros(k, k);
    for j in 0..k {
        let step = 1e-5 * th[j].abs().max(1e-2);
        let mut a = th.to_vec();
        let mut b = th.to_vec();
        a[j] += step;
        b[j] -= step;
        let (ga, gb) = (g(&a), g(&b));
        for i in 0..k {
            h[(i, j)] = (ga[i] - gb[i]) / (2.0 * step);
        }
    }
    for i in 0..k {
        for j in i + 1..k {
            let v = 0.5 * (h[(i, j)] + h[(j, i)]);
            h[(i, j)] = v;
            h[(j, i)] = v;
        }
    }
    h
}

/// Hessian of a function by second differences.
fn num_hessian_f(th: &[f64], f: &dyn Fn(&[f64]) -> f64) -> Matrix {
    let k = th.len();
    let h: Vec<f64> = th.iter().map(|v| 1e-4 * v.abs().max(1e-2)).collect();
    let mut m = Matrix::zeros(k, k);
    let f0 = f(th);
    for i in 0..k {
        for j in i..k {
            let v = if i == j {
                let mut a = th.to_vec();
                let mut b = th.to_vec();
                a[i] += h[i];
                b[i] -= h[i];
                (f(&a) - 2.0 * f0 + f(&b)) / (h[i] * h[i])
            } else {
                let mut s = 0.0;
                for (si, sj, sg) in [(1.0, 1.0, 1.0), (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0)] {
                    let mut a = th.to_vec();
                    a[i] += si * h[i];
                    a[j] += sj * h[j];
                    s += sg * f(&a);
                }
                s / (4.0 * h[i] * h[j])
            };
            m[(i, j)] = v;
            m[(j, i)] = v;
        }
    }
    m
}

/// Golden-section maximization on [a, b].
fn golden_max(f: &dyn Fn(f64) -> f64, mut a: f64, mut b: f64) -> f64 {
    // coarse grid first
    let ng = 40;
    let mut best = (f64::NEG_INFINITY, a);
    for k in 0..=ng {
        let x = a + (b - a) * k as f64 / ng as f64;
        let v = f(x);
        if v > best.0 {
            best = (v, x);
        }
    }
    let step = (b - a) / ng as f64;
    a = (best.1 - step).max(a);
    b = (best.1 + step).min(b);
    let g = (5f64.sqrt() - 1.0) / 2.0;
    let mut c = b - g * (b - a);
    let mut d = a + g * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..200 {
        if (b - a).abs() < 1e-12 {
            break;
        }
        if fc > fd {
            b = d;
            d = c;
            fd = fc;
            c = b - g * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + g * (b - a);
            fd = f(d);
        }
    }
    (a + b) / 2.0
}

/// Nelder–Mead maximization.
fn nelder_mead(f: &dyn Fn(&[f64]) -> f64, x0: &[f64], scale: f64, iters: usize) -> Vec<f64> {
    let k = x0.len();
    let mut pts: Vec<Vec<f64>> = vec![x0.to_vec()];
    for i in 0..k {
        let mut p = x0.to_vec();
        p[i] += scale;
        pts.push(p);
    }
    let mut vals: Vec<f64> = pts.iter().map(|p| -f(p)).collect();
    for _ in 0..iters {
        let mut idx: Vec<usize> = (0..=k).collect();
        idx.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
        pts = idx.iter().map(|&i| pts[i].clone()).collect();
        vals = idx.iter().map(|&i| vals[i]).collect();
        if (vals[k] - vals[0]).abs() < 1e-13 * (1.0 + vals[0].abs()) {
            let spread = pts.iter().map(|p| p.iter().zip(&pts[0]).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)).fold(0.0, f64::max);
            if spread < 1e-10 {
                break;
            }
        }
        let cen: Vec<f64> = (0..k).map(|j| pts[..k].iter().map(|p| p[j]).sum::<f64>() / k as f64).collect();
        let at = |t: f64| -> Vec<f64> { (0..k).map(|j| cen[j] + t * (pts[k][j] - cen[j])).collect() };
        let xr = at(-1.0);
        let fr = -f(&xr);
        if fr < vals[0] {
            let xe = at(-2.0);
            let fe = -f(&xe);
            if fe < fr {
                pts[k] = xe;
                vals[k] = fe;
            } else {
                pts[k] = xr;
                vals[k] = fr;
            }
        } else if fr < vals[k - 1] {
            pts[k] = xr;
            vals[k] = fr;
        } else {
            let xc = if fr < vals[k] { at(-0.5) } else { at(0.5) };
            let fc = -f(&xc);
            if fc < vals[k].min(fr) {
                pts[k] = xc;
                vals[k] = fc;
            } else {
                for i in 1..=k {
                    pts[i] = (0..k).map(|j| pts[0][j] + 0.5 * (pts[i][j] - pts[0][j])).collect();
                    vals[i] = -f(&pts[i]);
                }
            }
        }
    }
    let best = (0..=k).min_by(|&a, &b| vals[a].total_cmp(&vals[b])).unwrap();
    pts[best].clone()
}

struct MlFit {
    theta: Vec<f64>,
    v: Matrix,
    ll: f64,
    conc_log: Vec<f64>,
    full_log: Vec<f64>,
    converged: bool,
}

fn fit_ml(m: &Ml, has_l: bool, has_r: bool, lb: (f64, f64), rb: (f64, f64)) -> Result<MlFit, String> {
    let shrink = |b: (f64, f64)| (b.0 + 1e-7 * (b.1 - b.0), b.1 - 1e-7 * (b.1 - b.0));
    let (lb, rb) = (shrink(lb), shrink(rb));
    let inside = |l: f64, r: f64| (!has_l || (l > lb.0 && l < lb.1)) && (!has_r || (r > rb.0 && r < rb.1));
    let conc = |l: f64, r: f64| if inside(l, r) { m.conc(l, r).0 } else { f64::NEG_INFINITY };
    let mut conc_log = vec![m.conc(0.0, 0.0).0];
    let (lam, rho) = match (has_l, has_r) {
        (true, false) => (golden_max(&|l| conc(l, 0.0), lb.0, lb.1), 0.0),
        (false, true) => (0.0, golden_max(&|r| conc(0.0, r), rb.0, rb.1)),
        (true, true) => {
            // alternate one-dimensional searches, then Nelder–Mead polish
            let (mut l, mut r) = (0.0, 0.0);
            for _ in 0..30 {
                let l2 = golden_max(&|x| conc(x, r), lb.0, lb.1);
                let r2 = golden_max(&|x| conc(l2, x), rb.0, rb.1);
                let done = (l2 - l).abs() + (r2 - r).abs() < 1e-10;
                l = l2;
                r = r2;
                conc_log.push(conc(l, r));
                if done {
                    break;
                }
            }
            let p = nelder_mead(&|p: &[f64]| conc(p[0], p[1]), &[l, r], 1e-3, 400);
            (p[0], p[1])
        }
        _ => (0.0, 0.0),
    };
    let (llc, beta, s2) = m.conc(lam, rho);
    conc_log.push(llc);
    let mut th = beta.clone();
    if has_l {
        th.push(lam);
    }
    if has_r {
        th.push(rho);
    }
    th.push(s2);
    let f = |t: &[f64]| {
        let k = m.kx();
        let ok = (!has_l || (t[k] > lb.0 && t[k] < lb.1)) && (!has_r || {
            let r = t[k + has_l as usize];
            r > rb.0 && r < rb.1
        });
        if ok { m.ll(t, has_l, has_r) } else { f64::NEG_INFINITY }
    };
    let g = |t: &[f64]| m.grad(t, has_l, has_r);
    let mut full_log = vec![f(&th)];
    let mut converged = false;
    for _ in 0..50 {
        let gr = g(&th);
        let h = num_hessian(&th, &g);
        let hi = inv(&h.scale(-1.0));
        let step = mat_vec(&hi, &gr);
        let crit = dot(&gr, &step).abs();
        if crit < 1e-14 {
            converged = true;
            break;
        }
        let cur = f(&th);
        let mut t = 1.0;
        let mut moved = false;
        for _ in 0..30 {
            let cand: Vec<f64> = th.iter().zip(&step).map(|(a, b)| a + t * b).collect();
            if f(&cand) >= cur - 1e-12 * cur.abs() {
                th = cand;
                moved = true;
                break;
            }
            t /= 2.0;
        }
        full_log.push(f(&th));
        if !moved || crit < 1e-12 {
            converged = true;
            break;
        }
    }
    let h = num_hessian(&th, &g);
    let v = inv(&h.scale(-1.0));
    Ok(MlFit { ll: f(&th), theta: th, v, conc_log, full_log, converged })
}

// ---------------------------------------------------------------------------
// GS2SLS
// ---------------------------------------------------------------------------

struct Gs2slsFit {
    delta: Vec<f64>,
    rho: Option<f64>,
    v: Matrix,
    sigma2: f64,
}

struct Moments {
    a: [Sparse; 2],
    b: [Sparse; 2], // A + A'
    vd: [Vec<f64>; 2],
}

fn moments_setup(m: &Sparse, het: bool) -> Moments {
    let n = m.n as f64;
    let mtm = m.t().mul(m);
    let a1 = if het {
        mtm.lin(1.0, &Sparse::from_diag(&mtm.diag()), -1.0)
    } else {
        let tr = mtm.trace() / n;
        let v = 1.0 / (1.0 + tr * tr);
        mtm.lin(v, &Sparse::identity(m.n), -v * tr)
    };
    let a2 = m.clone();
    let b1 = a1.lin(1.0, &a1.t(), 1.0);
    let b2 = a2.lin(1.0, &a2.t(), 1.0);
    let vd = [a1.diag(), a2.diag()];
    Moments { a: [a1, a2], b: [b1, b2], vd }
}

/// γ (2) and Γ (2×2) of the moment conditions m(ρ) = γ − Γ (ρ, ρ²)'.
fn gamma(mo: &Moments, m: &Sparse, u: &[f64]) -> ([f64; 2], [[f64; 2]; 2]) {
    let n = u.len() as f64;
    let ub = m.mv(u);
    let mut g = [0.0; 2];
    let mut gg = [[0.0; 2]; 2];
    for s in 0..2 {
        g[s] = mo.a[s].quad(u, u) / n;
        gg[s][0] = mo.b[s].quad(u, &ub) / n;
        gg[s][1] = -mo.a[s].quad(&ub, &ub) / n;
    }
    (g, gg)
}

fn gmm_rho(g: &[f64; 2], gg: &[[f64; 2]; 2], wt: &[[f64; 2]; 2]) -> f64 {
    let obj = |r: f64| {
        let m = [g[0] - gg[0][0] * r - gg[0][1] * r * r, g[1] - gg[1][0] * r - gg[1][1] * r * r];
        -(m[0] * (wt[0][0] * m[0] + wt[0][1] * m[1]) + m[1] * (wt[1][0] * m[0] + wt[1][1] * m[1]))
    };
    golden_max(&obj, -0.9999, 0.9999)
}

/// Ψ_ρρ (2×2), the a_r vectors, and the moment variance pieces.
#[allow(clippy::too_many_arguments)]
fn psi_rr(mo: &Moments, m: &Sparse, h: &[Vec<f64>], z: &[Vec<f64>], eps: &[f64], rho: f64, het: bool) -> ([[f64; 2]; 2], [Vec<f64>; 2], Matrix) {
    let n = eps.len();
    let nf = n as f64;
    // Z* = (I − ρM) Z, P = Q_HH^{-1} Q_HZ* [Q_HZ*' Q_HH^{-1} Q_HZ*]^{-1}
    let zs: Vec<Vec<f64>> = z.iter().map(|c| c.iter().zip(m.mv(c)).map(|(a, b)| a - rho * b).collect()).collect();
    let qhh = cross(h, h).scale(1.0 / nf);
    let qhz = cross(h, &zs).scale(1.0 / nf);
    let qhhi = inv(&qhh);
    let mid = inv(&mmul(&mmul(&qhz.transpose(), &qhhi), &qhz));
    let p = mmul(&mmul(&qhhi, &qhz), &mid);
    let mut avec: [Vec<f64>; 2] = [vec![0.0; n], vec![0.0; n]];
    for s in 0..2 {
        let be = mo.b[s].mv(eps);
        let alpha: Vec<f64> = zs.iter().map(|c| -dot(c, &be) / nf).collect();
        let pa = mat_vec(&p, &alpha);
        avec[s] = comb(h, &pa, n);
    }
    let mut psi = [[0.0; 2]; 2];
    if het {
        let s2: Vec<f64> = eps.iter().map(|e| e * e).collect();
        for r in 0..2 {
            for s in 0..2 {
                let tr = weighted_frob(&mo.b[r], &mo.b[s], Some(&s2));
                let aa: f64 = (0..n).map(|i| avec[r][i] * avec[s][i] * s2[i]).sum();
                psi[r][s] = tr / (2.0 * nf) + aa / nf;
            }
        }
    } else {
        let s2 = dot(eps, eps) / nf;
        let mu3 = eps.iter().map(|e| e.powi(3)).sum::<f64>() / nf;
        let mu4 = eps.iter().map(|e| e.powi(4)).sum::<f64>() / nf;
        for r in 0..2 {
            for s in 0..2 {
                let tr = weighted_frob(&mo.b[r], &mo.b[s], None);
                psi[r][s] = s2 * s2 * tr / (2.0 * nf) + s2 * dot(&avec[r], &avec[s]) / nf + (mu4 - 3.0 * s2 * s2) * dot(&mo.vd[r], &mo.vd[s]) / nf + mu3 * (dot(&avec[r], &mo.vd[s]) + dot(&avec[s], &mo.vd[r])) / nf;
            }
        }
    }
    (psi, avec, p)
}

fn inv2(a: &[[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let d = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    [[a[1][1] / d, -a[0][1] / d], [-a[1][0] / d, a[0][0] / d]]
}

fn twosls(y: &[f64], z: &[Vec<f64>], h: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = y.len();
    let hhi = inv(&cross(h, h));
    let hz = cross(h, z);
    let coef = mmul(&hhi, &hz); // kH × kz
    let zhat: Vec<Vec<f64>> = (0..z.len()).map(|j| comb(h, &(0..h.len()).map(|r| coef[(r, j)]).collect::<Vec<_>>(), n)).collect();
    let a = inv(&cross(&zhat, z));
    let zy: Vec<f64> = zhat.iter().map(|c| dot(c, y)).collect();
    (mat_vec(&a, &zy), zhat)
}

fn fit_gs2sls(y: &[f64], z: &[Vec<f64>], h: &[Vec<f64>], mlag: Option<&Sparse>, het: bool) -> Gs2slsFit {
    let n = y.len();
    let nf = n as f64;
    let (d1, zhat) = twosls(y, z, h);
    let u1: Vec<f64> = (0..n).map(|i| y[i] - comb(z, &d1, n)[i]).collect::<Vec<_>>();
    let Some(m) = mlag else {
        let zz = inv(&cross(&zhat, &zhat));
        let s2 = dot(&u1, &u1) / nf;
        let v = if het {
            let kz = z.len();
            let mut meat = Matrix::zeros(kz, kz);
            for i in 0..n {
                for a in 0..kz {
                    for b in 0..kz {
                        meat[(a, b)] += zhat[a][i] * zhat[b][i] * u1[i] * u1[i];
                    }
                }
            }
            mmul(&mmul(&zz, &meat), &zz)
        } else {
            zz.scale(s2)
        };
        return Gs2slsFit { delta: d1, rho: None, v, sigma2: s2 };
    };
    let mo = moments_setup(m, het);
    // step 1b: initial GMM
    let (g, gg) = gamma(&mo, m, &u1);
    let rho0 = gmm_rho(&g, &gg, &[[1.0, 0.0], [0.0, 1.0]]);
    // step 1c: efficient GMM
    let e1: Vec<f64> = u1.iter().zip(m.mv(&u1)).map(|(a, b)| a - rho0 * b).collect();
    // P built from the untransformed Z in the first step
    let (psi, _, _) = psi_rr(&mo, m, h, z, &e1, 0.0, het);
    let rho1 = gmm_rho(&g, &gg, &inv2(&psi));
    // step 2a: GS2SLS on the Cochrane–Orcutt transformed model
    let ys: Vec<f64> = y.iter().zip(m.mv(y)).map(|(a, b)| a - rho1 * b).collect();
    let zs: Vec<Vec<f64>> = z.iter().map(|c| c.iter().zip(m.mv(c)).map(|(a, b)| a - rho1 * b).collect()).collect();
    let (d2, _) = twosls(&ys, &zs, h);
    let u2: Vec<f64> = {
        let zd = comb(z, &d2, n);
        (0..n).map(|i| y[i] - zd[i]).collect()
    };
    // step 2b: efficient GMM with Ψ(ρ̃, δ̂)
    let (g2, gg2) = gamma(&mo, m, &u2);
    let e2: Vec<f64> = u2.iter().zip(m.mv(&u2)).map(|(a, b)| a - rho1 * b).collect();
    let (psi2, _, _) = psi_rr(&mo, m, h, z, &e2, rho1, het);
    let rho = gmm_rho(&g2, &gg2, &inv2(&psi2));
    // joint variance at (δ̂, ρ̂)
    let eps: Vec<f64> = u2.iter().zip(m.mv(&u2)).map(|(a, b)| a - rho * b).collect();
    let (psi_f, avec, p) = psi_rr(&mo, m, h, z, &eps, rho, het);
    let kh = h.len();
    let kz = z.len();
    let s2 = dot(&eps, &eps) / nf;
    let psi_dd = if het {
        let mut a = Matrix::zeros(kh, kh);
        for i in 0..n {
            let w = eps[i] * eps[i];
            for r in 0..kh {
                for c in 0..kh {
                    a[(r, c)] += h[r][i] * h[c][i] * w;
                }
            }
        }
        a.scale(1.0 / nf)
    } else {
        cross(h, h).scale(s2 / nf)
    };
    let mut psi_dr = Matrix::zeros(kh, 2);
    let mu3 = eps.iter().map(|e| e.powi(3)).sum::<f64>() / nf;
    for r in 0..kh {
        for s in 0..2 {
            psi_dr[(r, s)] = if het {
                (0..n).map(|i| h[r][i] * eps[i] * eps[i] * avec[s][i]).sum::<f64>() / nf
            } else {
                s2 * dot(&h[r], &avec[s]) / nf + mu3 * dot(&h[r], &mo.vd[s]) / nf
            };
        }
    }
    let jv = [gg2[0][0] + 2.0 * rho * gg2[0][1], gg2[1][0] + 2.0 * rho * gg2[1][1]];
    let pi = inv2(&psi_f);
    let jpj = jv[0] * (pi[0][0] * jv[0] + pi[0][1] * jv[1]) + jv[1] * (pi[1][0] * jv[0] + pi[1][1] * jv[1]);
    // row vector L = (J'Ψ⁻¹J)^{-1} J'Ψ⁻¹
    let l = [(jv[0] * pi[0][0] + jv[1] * pi[1][0]) / jpj, (jv[0] * pi[0][1] + jv[1] * pi[1][1]) / jpj];
    // Ω = [P' 0; 0 L] [Ψdd Ψdr; Ψdr' Ψrr] [P 0; 0 L']
    let vdd = mmul(&mmul(&p.transpose(), &psi_dd), &p);
    let pdr = mmul(&p.transpose(), &psi_dr); // kz × 2
    let vdr: Vec<f64> = (0..kz).map(|a| pdr[(a, 0)] * l[0] + pdr[(a, 1)] * l[1]).collect();
    let vrr = l[0] * (psi_f[0][0] * l[0] + psi_f[0][1] * l[1]) + l[1] * (psi_f[1][0] * l[0] + psi_f[1][1] * l[1]);
    let mut v = Matrix::zeros(kz + 1, kz + 1);
    for a in 0..kz {
        for b in 0..kz {
            v[(a, b)] = vdd[(a, b)] / nf;
        }
        v[(a, kz)] = vdr[a] / nf;
        v[(kz, a)] = vdr[a] / nf;
    }
    v[(kz, kz)] = vrr / nf;
    Gs2slsFit { delta: d2, rho: Some(rho), v, sigma2: s2 }
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

fn head_line(left: &str, label: &str, val: &str) -> String {
    format!("{:<48}{:<17} = {:>10}", left, label, val)
}

fn place_notes(p: &Prep) -> Vec<String> {
    let mut out = vec![format!("  ({} observations)", p.nobs_all)];
    if p.t > 1 {
        out.push(format!("  ({} observations used)", p.n * p.t));
        out.push(format!("  ({} places used)", p.n));
    } else {
        out.push(format!("  ({} observations (places) used)", p.n));
    }
    let mut names: Vec<&String> = p.w_places.keys().collect();
    names.sort();
    for nm in names {
        out.push(format!("  (weighting matrix {nm} defines {} places)", p.w_places[nm]));
    }
    out
}

fn table(depvar: &str, names: &[String], b: &[f64], v: &Matrix, level: f64) -> Vec<String> {
    let k = names.len();
    let omitted = vec![false; k];
    let is_base = vec![false; k];
    let lines = est::coef_table(&est::TableSpec { depvar, names, b, v, omitted: &omitted, is_base: &is_base, df: None, level, eform: None, coef_title: "Coefficient" });
    lines.into_iter().map(|l| l.replacen("/var(e.", " var(e.", 1)).collect()
}

#[allow(clippy::too_many_arguments)]
fn post_results(st: &mut Store, cmd: &Command, verb: &str, p: &Prep, sp: &Spec, names: &[String], b: &[f64], v: &Matrix, scalars: &[(&str, f64)], macros: &[(&str, String)]) {
    let e = st.begin_e();
    e.set("N", (p.n * p.t) as f64);
    e.set("rank", b.len() as f64);
    for (k, x) in scalars {
        e.set(k, *x);
    }
    e.set_str("cmd", verb);
    e.set_str("cmdline", format!("{} {}", verb, cmd.raw.split_once(' ').map(|x| x.1).unwrap_or("")).trim().to_string());
    e.set_str("depvar", sp.depvar.clone());
    e.set_str("indeps", p.xnames.iter().filter(|n| *n != "_cons").cloned().collect::<Vec<_>>().join(" "));
    e.set_str("title", "Spatial autoregressive model");
    e.set_str("properties", "b V");
    e.set_str("estat_cmd", "spregress_estat");
    e.set_str("predict", "spregress_p");
    e.set_str("constant", if p.has_cons { "hasconstant" } else { "noconstant" });
    e.set_str("dlmat", sp.dl.clone().unwrap_or_default());
    e.set_str("elmat", sp.el.clone().unwrap_or_default());
    let ivm: Vec<String> = sp.iv.iter().map(|x| x.0.clone()).collect();
    e.set_str("ivlmat", ivm.join(" "));
    e.set_str("idvar", "_ID");
    e.set_str("exogr", sp.inst.join(" "));
    e.set_str("endog", sp.endog.join(" "));
    for (k, x) in macros {
        e.set_str(k, x.clone());
    }
    e.set_mat("b", Matrix::row_vector(b).with_names(vec!["y1".into()], names.to_vec()));
    e.set_mat("V", v.clone().with_names(names.to_vec(), names.to_vec()));
    e.sample = Some(p.mask.clone());
    let _ = p.nobs_all;
}

fn spatial_pick(depvar: &str) -> impl Fn(&str) -> bool + '_ {
    move |n: &str| {
        let (eq, _) = n.split_once(':').unwrap_or(("", n));
        eq != depvar && eq != "/" && !eq.starts_with("var(")
    }
}

fn x_pick(depvar: &str) -> impl Fn(&str) -> bool + '_ {
    move |n: &str| {
        let (eq, v) = n.split_once(':').unwrap_or(("", n));
        eq == depvar && v != "_cons"
    }
}

fn wald_lines(names: &[String], b: &[f64], v: &Matrix, depvar: &str) -> (Vec<String>, f64, usize) {
    let (c, df) = wald(names, b, v, &spatial_pick(depvar));
    if df == 0 {
        return (Vec::new(), 0.0, 0);
    }
    let p = dist::chi2tail(df as f64, c);
    (vec![format!("Wald test of spatial terms:          chi2({df}) = {:<10}Prob > chi2 = {:.4}", format!("{:.2}", c), p)], c, df)
}

// ---------------------------------------------------------------------------
// spregress / spivregress
// ---------------------------------------------------------------------------

pub struct Spregress {
    pub iv: bool,
}

impl CommandHandler for Spregress {
    fn name(&self) -> &'static str {
        if self.iv { "spivregress" } else { "spregress" }
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let verb = self.name();
        let sp = parse_spec(ds, cmd, self.iv)?;
        let ml = cmd.has_opt("ml", 2);
        let gs = cmd.has_opt("gs2sls", 2);
        if self.iv && ml {
            return Err(rc_err(198, "spivregress fits by GS2SLS only"));
        }
        if !self.iv && ml == gs {
            return Err(rc_err(198, "must specify either gs2sls or ml"));
        }
        if ml && cmd.has_opt("heteroskedastic", 3) {
            return Err(rc_err(198, "option heteroskedastic not allowed with ml"));
        }
        let p = prepare(ds, st, cmd, &sp, false)?;
        let level = cmd.opt("level", 1).and_then(|s| s.trim().parse().ok()).unwrap_or(st.level);
        let n = p.n;
        let y = &p.y;
        let mut out = place_notes(&p);
        out.push(String::new());
        let wl = sp.dl.as_ref().map(|w| p.mats[w].clone());
        let ml_m = sp.el.as_ref().map(|w| p.mats[w].clone());
        let het = cmd.has_opt("heteroskedastic", 3);
        let (names, b, v, ll, s2, rho_hat, lam_hat, extra_s): (Vec<String>, Vec<f64>, Matrix, Option<f64>, f64, f64, f64, Vec<(&str, f64)>);
        if ml {
            let ew = wl.as_ref().map(|w| eigenvalues(&sparse_dense(w)));
            let em = ml_m.as_ref().map(|m| eigenvalues(&sparse_dense(m)));
            let mut x = p.xcols.clone();
            x.extend(p.lags.iter().map(|l| l.2.clone()));
            let wy = wl.as_ref().map(|w| w.mv(y)).unwrap_or_else(|| vec![0.0; n]);
            let mm = ml_m.clone().unwrap_or(Sparse { n, rows: vec![Vec::new(); n] });
            let model = Ml { y: y.clone(), my: mm.mv(y), mwy: mm.mv(&wy), wy, mx: x.iter().map(|c| mm.mv(c)).collect(), x, ew: ew.as_deref(), em: em.as_deref(), tf: 1.0, neff: n as f64 };
            let lb = ew.as_ref().map(|e| param_bounds(e)).unwrap_or((-1.0, 1.0));
            let rb = em.as_ref().map(|e| param_bounds(e)).unwrap_or((-1.0, 1.0));
            let fit = fit_ml(&model, wl.is_some(), ml_m.is_some(), lb, rb)?;
            let lay = layout(&p, &sp, &[format!("/:var(e.{})", sp.depvar)]);
            let (bb, vv) = reorder(&lay, &fit.theta, &fit.v);
            let k = p.xcols.len() + p.lags.len();
            lam_hat = if wl.is_some() { fit.theta[k] } else { 0.0 };
            rho_hat = if ml_m.is_some() { fit.theta[k + wl.is_some() as usize] } else { 0.0 };
            s2 = *fit.theta.last().unwrap();
            if wl.is_some() || ml_m.is_some() {
                out.push("Optimizing concentrated log likelihood:".into());
                out.push(String::new());
                out.extend(est::iteration_log(&fit.conc_log, "Log likelihood"));
                out.push(String::new());
            }
            out.push("Optimizing unconcentrated log likelihood:".into());
            out.push(String::new());
            out.extend(est::iteration_log(&fit.full_log, "Log likelihood"));
            out.push(String::new());
            ll = Some(fit.ll);
            names = lay.names;
            b = bb;
            v = vv;
            extra_s = vec![("converged", if fit.converged { 1.0 } else { 0.0 }), ("iterations", (fit.full_log.len() - 1) as f64)];
        } else {
            // Z = [X, WX lags, endog, Wy]
            let mut z = p.xcols.clone();
            z.extend(p.lags.iter().map(|l| l.2.clone()));
            z.extend(p.endog.iter().map(|e| e.1.clone()));
            if let Some(w) = &wl {
                z.push(w.mv(y));
            }
            // instruments
            let q: usize = cmd.opt("impower", 3).and_then(|s| s.trim().parse().ok()).unwrap_or(2);
            let mut base: Vec<Vec<f64>> = p.xcols.clone();
            base.extend(p.lags.iter().map(|l| l.2.clone()));
            base.extend(p.inst.iter().cloned());
            let nonconst: Vec<Vec<f64>> = base.iter().filter(|c| c.iter().any(|v| *v != c[0])).cloned().collect();
            let mut hcols = base.clone();
            let mut powers: Vec<Vec<f64>> = Vec::new();
            if let Some(w) = &wl {
                let mut cur = nonconst.clone();
                for _ in 0..q {
                    cur = cur.iter().map(|c| w.mv(c)).collect();
                    powers.extend(cur.iter().cloned());
                }
            } else if !p.endog.is_empty() {
                for (_, w) in p.mats.iter() {
                    powers.extend(nonconst.iter().map(|c| w.mv(c)));
                }
            }
            hcols.extend(powers);
            if let Some(m) = &ml_m {
                let extra: Vec<Vec<f64>> = hcols.iter().filter(|c| c.iter().any(|v| *v != c[0])).map(|c| m.mv(c)).collect();
                hcols.extend(extra);
            }
            let h = independent(hcols);
            if h.len() < z.len() {
                return Err(rc_err(481, "equation not identified; too few instruments"));
            }
            let fit = fit_gs2sls(y, &z, &h, ml_m.as_ref(), het);
            // internal order: x, lags, endog, λ?, ρ?
            let mut theta = fit.delta.clone();
            let mut vfull = fit.v.clone();
            if fit.rho.is_none() {
                // nothing to add
            } else {
                theta.push(fit.rho.unwrap());
            }
            if theta.len() != vfull.rows {
                vfull = Matrix::zeros(theta.len(), theta.len());
            }
            let lay = layout(&p, &sp, &[]);
            let (bb, vv) = reorder(&lay, &theta, &vfull);
            let kz = z.len();
            lam_hat = if wl.is_some() { fit.delta[kz - 1] } else { 0.0 };
            rho_hat = fit.rho.unwrap_or(0.0);
            s2 = fit.sigma2;
            ll = None;
            names = lay.names;
            b = bb;
            v = vv;
            extra_s = vec![];
        }
        // pseudo R²: squared correlation of y with the reduced-form prediction
        let bmap: HashMap<&str, f64> = names.iter().map(|s| s.as_str()).zip(b.iter().cloned()).collect();
        let mut xb = vec![0.0; n];
        for (k, nm) in p.xnames.iter().enumerate() {
            let c = bmap.get(format!("{}:{nm}", sp.depvar).as_str()).copied().unwrap_or(0.0);
            for i in 0..n {
                xb[i] += c * p.xcols[k][i];
            }
        }
        for (wn, vn, col) in &p.lags {
            let c = bmap.get(format!("{wn}:{vn}").as_str()).copied().unwrap_or(0.0);
            for i in 0..n {
                xb[i] += c * col[i];
            }
        }
        for (en, col) in &p.endog {
            let c = bmap.get(format!("{}:{en}", sp.depvar).as_str()).copied().unwrap_or(0.0);
            for i in 0..n {
                xb[i] += c * col[i];
            }
        }
        let rform = match &wl {
            Some(w) => solve_ilw(w, lam_hat, &xb),
            None => xb.clone(),
        };
        let r2p = squared_corr(y, &rform);
        let (chi2, df_m) = wald(&names, &b, &v, &x_pick(&sp.depvar));
        let pchi = dist::chi2tail(df_m as f64, chi2);
        let title = "Spatial autoregressive model";
        let sub = if ml { "Maximum likelihood estimates" } else { "GS2SLS estimates" };
        out.push(head_line(title, "Number of obs", &util::commas(n)));
        out.push(head_line(sub, &format!("Wald chi2({df_m})"), &format!("{:.2}", chi2)));
        out.push(head_line("", "Prob > chi2", &format!("{:.4}", pchi)));
        match ll {
            Some(l) => out.push(head_line(&format!("Log likelihood = {}", fmt_g(l, 10)), "Pseudo R2", &format!("{:.4}", r2p))),
            None => out.push(head_line("", "Pseudo R2", &format!("{:.4}", r2p))),
        }
        out.push(String::new());
        out.extend(table(&sp.depvar, &names, &b, &v, level));
        let (wl_lines, chi2_c, df_c) = wald_lines(&names, &b, &v, &sp.depvar);
        out.extend(wl_lines);
        let mut scal: Vec<(&str, f64)> = vec![("df_m", df_m as f64), ("chi2", chi2), ("p", pchi), ("r2_p", r2p), ("df_c", df_c as f64), ("chi2_c", chi2_c), ("p_c", dist::chi2tail(df_c as f64, chi2_c)), ("k", b.len() as f64), ("sigma2", s2)];
        if let Some(l) = ll {
            scal.push(("ll", l));
        }
        if sp.dl.is_some() {
            scal.push(("lambda", lam_hat));
        }
        if sp.el.is_some() {
            scal.push(("rho", rho_hat));
        }
        scal.extend(extra_s);
        let mut macros = vec![("estimator", if ml { "ml".to_string() } else { "gs2sls".to_string() }), ("het", if het { "heteroskedastic".into() } else { "homoskedastic".into() })];
        macros.push(("model", match (sp.dl.is_some() || !sp.iv.is_empty(), sp.el.is_some()) {
            (true, true) => "sarar".into(),
            (true, false) => "sar".into(),
            (false, true) => "sem".into(),
            _ => "lr".into(),
        }));
        macros.push(("vce", if ml { "oim".into() } else if het { "robust".into() } else { "unadjusted".into() }));
        post_results(st, cmd, verb, &p, &sp, &names, &b, &v, &scal, &macros);
        Ok(out.join("\n"))
    }
}

fn sparse_dense(s: &Sparse) -> Matrix {
    let mut m = Matrix::zeros(s.n, s.n);
    for (i, r) in s.rows.iter().enumerate() {
        for &(j, v) in r {
            m[(i, j)] = v;
        }
    }
    m
}

// ---------------------------------------------------------------------------
// spxtregress
// ---------------------------------------------------------------------------

pub struct Spxtregress;

impl CommandHandler for Spxtregress {
    fn name(&self) -> &'static str {
        "spxtregress"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let sp = parse_spec(ds, cmd, false)?;
        let fe = cmd.has_opt("fe", 2);
        let re = cmd.has_opt("re", 2);
        if fe == re {
            return Err(rc_err(198, "must specify either fe or re"));
        }
        let sarpanel = cmd.has_opt("sarpanel", 4);
        if re && sp.el.is_some() && !sarpanel {
            return Err(rc_err(198, "errorlag() with re requires option sarpanel in this implementation"));
        }
        let p = prepare(ds, st, cmd, &sp, true)?;
        let level = cmd.opt("level", 1).and_then(|s| s.trim().parse().ok()).unwrap_or(st.level);
        let (n, t) = (p.n, p.t);
        if t < 2 {
            return Err(rc_err(459, "at least two time periods required"));
        }
        let nt = n * t;
        let wl = sp.dl.as_ref().map(|w| p.mats[w].clone());
        let mm = sp.el.as_ref().map(|w| p.mats[w].clone());
        let ew = wl.as_ref().map(|w| eigenvalues(&sparse_dense(w)));
        let em = mm.as_ref().map(|m| eigenvalues(&sparse_dense(m)));
        let lb = ew.as_ref().map(|e| param_bounds(e)).unwrap_or((-1.0, 1.0));
        let rb = em.as_ref().map(|e| param_bounds(e)).unwrap_or((-1.0, 1.0));
        let mut out = place_notes(&p);
        out.push(String::new());
        let y = p.y.clone();
        let mut x: Vec<Vec<f64>> = p.xcols.clone();
        x.extend(p.lags.iter().map(|l| l.2.clone()));
        let mut xnames_int: Vec<String> = p.xnames.clone();
        xnames_int.extend(p.lags.iter().map(|l| format!("{}:{}", l.0, l.1)));
        let wy = wl.as_ref().map(|w| w.mv_blocks(&y)).unwrap_or_else(|| vec![0.0; nt]);
        let zero = Sparse { n, rows: vec![Vec::new(); n] };
        let mmat = mm.clone().unwrap_or(zero);
        let has_l = wl.is_some();
        let has_r = mm.is_some();
        let (names, b, v, ll, conc_log, full_log, converged);
        if fe {
            // within transformation; Lee–Yu likelihood with n(T−1) effective observations
            let demean = |v: &[f64]| -> Vec<f64> {
                let mut m = vec![0.0; n];
                for pdx in 0..t {
                    for i in 0..n {
                        m[i] += v[pdx * n + i] / t as f64;
                    }
                }
                (0..nt).map(|k| v[k] - m[k % n]).collect()
            };
            let keep: Vec<usize> = (0..x.len()).filter(|&j| xnames_int[j] != "_cons").collect();
            let xd: Vec<Vec<f64>> = keep.iter().map(|&j| demean(&x[j])).collect();
            let keep_names: Vec<String> = keep.iter().map(|&j| xnames_int[j].clone()).collect();
            let yd = demean(&y);
            let wyd = demean(&wy);
            let model = Ml { my: mmat.mv_blocks(&yd), mwy: mmat.mv_blocks(&wyd), wy: wyd, y: yd, mx: xd.iter().map(|c| mmat.mv_blocks(c)).collect(), x: xd, ew: ew.as_deref(), em: em.as_deref(), tf: (t - 1) as f64, neff: (n * (t - 1)) as f64 };
            let fit = fit_ml(&model, has_l, has_r, lb, rb)?;
            // report σ_e (delta method from σ²)
            let mut th = fit.theta.clone();
            let mut vv = fit.v.clone();
            let k = th.len();
            let s2 = th[k - 1];
            let se = s2.sqrt();
            th[k - 1] = se;
            let dd = 0.5 / se;
            for a in 0..k {
                vv[(a, k - 1)] *= dd;
                vv[(k - 1, a)] *= dd;
            }
            let (lay_names, idx) = xt_layout(&keep_names, &sp, has_l, has_r, &["/:sigma_e".to_string()]);
            let (bb, v2) = reorder(&Layout { names: lay_names.clone(), idx }, &th, &vv);
            names = lay_names;
            b = bb;
            v = v2;
            ll = fit.ll;
            conc_log = fit.conc_log;
            full_log = fit.full_log;
            converged = fit.converged;
        } else {
            // random effects: quasi-demeaning with θ = 1 − sqrt(σe²/(σe² + Tσu²))
            let tt = t as f64;
            let bm = |v: &[f64], rho: f64| -> Vec<f64> {
                if has_r {
                    let mv = mmat.mv_blocks(v);
                    v.iter().zip(mv).map(|(a, b)| a - rho * b).collect()
                } else {
                    v.to_vec()
                }
            };
            let qd = |v: &[f64], th: f64| -> Vec<f64> {
                let mut m = vec![0.0; n];
                for pdx in 0..t {
                    for i in 0..n {
                        m[i] += v[pdx * n + i] / tt;
                    }
                }
                (0..nt).map(|k| v[k] - th * m[k % n]).collect()
            };
            let ld = |l: f64, r: f64| tt * (ew.as_ref().map_or(0.0, |e| logdet(e, l)) + em.as_ref().map_or(0.0, |e| logdet(e, r)));
            let ntf = nt as f64;
            // full log likelihood in (β, λ, ρ, σu, σe)
            let full = |th: &[f64]| -> f64 {
                let k = x.len();
                let lam = if has_l { th[k] } else { 0.0 };
                let rho = if has_r { th[k + has_l as usize] } else { 0.0 };
                let (su, se) = (th[th.len() - 2], th[th.len() - 1]);
                if se <= 0.0 || (has_l && !(lam > lb.0 && lam < lb.1)) || (has_r && !(rho > rb.0 && rho < rb.1)) {
                    return f64::NEG_INFINITY;
                }
                let r: Vec<f64> = {
                    let xb = comb(&x, &th[..k], nt);
                    (0..nt).map(|i| y[i] - lam * wy[i] - xb[i]).collect()
                };
                let rb_ = bm(&r, rho);
                let s1 = se * se;
                let s2 = s1 + tt * su * su;
                let th_q = 1.0 - (s1 / s2).sqrt();
                let q = qd(&rb_, th_q);
                let ssr = dot(&q, &q);
                -0.5 * ntf * (2.0 * std::f64::consts::PI).ln() - 0.5 * (n as f64) * (tt - 1.0) * s1.ln() - 0.5 * (n as f64) * s2.ln() + ld(lam, rho) - ssr / (2.0 * s1)
            };
            let conc = |l: f64, r: f64, thq: f64| -> (f64, Vec<f64>, f64) {
                if !(0.0..1.0).contains(&thq) || (has_l && !(l > lb.0 && l < lb.1)) || (has_r && !(r > rb.0 && r < rb.1)) {
                    return (f64::NEG_INFINITY, vec![], 0.0);
                }
                let yy: Vec<f64> = (0..nt).map(|i| y[i] - l * wy[i]).collect();
                let yq = qd(&bm(&yy, r), thq);
                let xq: Vec<Vec<f64>> = x.iter().map(|c| qd(&bm(c, r), thq)).collect();
                let beta = mat_vec(&inv(&cross(&xq, &xq)), &xq.iter().map(|c| dot(c, &yq)).collect::<Vec<_>>());
                let fit = comb(&xq, &beta, nt);
                let ssr: f64 = yq.iter().zip(&fit).map(|(a, b)| (a - b).powi(2)).sum();
                let s1 = ssr / ntf;
                let ll = -0.5 * ntf * ((2.0 * std::f64::consts::PI * s1).ln() + 1.0) + (n as f64) * (1.0 - thq).ln() + ld(l, r);
                (ll, beta, s1)
            };
            let mut start = vec![0.5];
            if has_l {
                start.insert(0, 0.0);
            }
            if has_r {
                start.insert(has_l as usize, 0.0);
            }
            let unpack = |pp: &[f64]| -> (f64, f64, f64) {
                let l = if has_l { pp[0] } else { 0.0 };
                let r = if has_r { pp[has_l as usize] } else { 0.0 };
                (l, r, pp[pp.len() - 1])
            };
            let fc = |pp: &[f64]| {
                let (l, r, q) = unpack(pp);
                conc(l, r, q).0
            };
            let mut clog = vec![fc(&start)];
            let mut pp = start.clone();
            for _ in 0..3 {
                pp = nelder_mead(&fc, &pp, 0.05, 2000);
                clog.push(fc(&pp));
            }
            let (l, r, q) = unpack(&pp);
            let (_, beta, s1) = conc(l, r, q);
            let s2 = s1 / (1.0 - q).powi(2);
            let su = ((s2 - s1) / tt).max(0.0).sqrt();
            let mut th = beta;
            if has_l {
                th.push(l);
            }
            if has_r {
                th.push(r);
            }
            th.push(su);
            th.push(s1.sqrt());
            let mut flog = vec![full(&th)];
            // Newton polish on the full likelihood with numerical derivatives
            let mut conv = false;
            for _ in 0..20 {
                let k = th.len();
                let g: Vec<f64> = (0..k)
                    .map(|j| {
                        let hh = 1e-6 * th[j].abs().max(1e-2);
                        let mut a = th.clone();
                        let mut c = th.clone();
                        a[j] += hh;
                        c[j] -= hh;
                        (full(&a) - full(&c)) / (2.0 * hh)
                    })
                    .collect();
                let h = num_hessian_f(&th, &full);
                let step = mat_vec(&inv(&h.scale(-1.0)), &g);
                let crit = dot(&g, &step).abs();
                if crit < 1e-10 {
                    conv = true;
                    break;
                }
                let cur = full(&th);
                let mut sz = 1.0;
                let mut moved = false;
                for _ in 0..30 {
                    let cand: Vec<f64> = th.iter().zip(&step).map(|(a, b)| a + sz * b).collect();
                    if full(&cand) >= cur {
                        th = cand;
                        moved = true;
                        break;
                    }
                    sz /= 2.0;
                }
                flog.push(full(&th));
                if !moved {
                    conv = true;
                    break;
                }
            }
            let h = num_hessian_f(&th, &full);
            let vv = inv(&h.scale(-1.0));
            let (lay_names, idx) = xt_layout(&xnames_int, &sp, has_l, has_r, &["/:sigma_u".to_string(), "/:sigma_e".to_string()]);
            let (bb, v2) = reorder(&Layout { names: lay_names.clone(), idx }, &th, &vv);
            names = lay_names;
            b = bb;
            v = v2;
            ll = full(&th);
            conc_log = clog;
            full_log = flog;
            converged = conv;
        }
        out.push("Optimizing concentrated log likelihood:".into());
        out.push(String::new());
        out.extend(est::iteration_log(&conc_log, "Log likelihood"));
        out.push(String::new());
        out.push("Optimizing unconcentrated log likelihood:".into());
        out.push(String::new());
        out.extend(est::iteration_log(&full_log, "Log likelihood"));
        out.push(String::new());
        let (chi2, df_m) = wald(&names, &b, &v, &x_pick(&sp.depvar));
        let pchi = dist::chi2tail(df_m as f64, chi2);
        let title = if fe { "Fixed-effects spatial regression" } else { "Random-effects spatial regression" };
        out.push(head_line(title, "Number of obs", &util::commas(nt)));
        out.push(head_line(&format!("Group variable: {}", p.panel_var.clone().unwrap_or_default()), "Number of groups", &util::commas(n)));
        out.push(head_line("", "Obs per group", &util::commas(t)));
        out.push(String::new());
        out.push(head_line("", &format!("Wald chi2({df_m})"), &format!("{:.2}", chi2)));
        out.push(head_line(&format!("Log likelihood = {}", fmt_g(ll, 10)), "Prob > chi2", &format!("{:.4}", pchi)));
        out.push(String::new());
        out.extend(table(&sp.depvar, &names, &b, &v, level));
        let (wl_lines, chi2_c, df_c) = wald_lines(&names, &b, &v, &sp.depvar);
        out.extend(wl_lines);
        let mut scal: Vec<(&str, f64)> = vec![("N_g", n as f64), ("T", t as f64), ("df_m", df_m as f64), ("chi2", chi2), ("p", pchi), ("ll", ll), ("df_c", df_c as f64), ("chi2_c", chi2_c), ("p_c", dist::chi2tail(df_c as f64, chi2_c)), ("converged", if converged { 1.0 } else { 0.0 }), ("k", b.len() as f64)];
        if let Some(i) = names.iter().position(|nm| nm == "/:sigma_e") {
            scal.push(("sigma_e", b[i]));
        }
        if let Some(i) = names.iter().position(|nm| nm == "/:sigma_u") {
            scal.push(("sigma_u", b[i]));
        }
        let macros = vec![("model", if fe { "fe".to_string() } else { "re".to_string() }), ("estimator", "ml".to_string()), ("vce", "oim".to_string()), ("ivar", p.panel_var.clone().unwrap_or_default())];
        post_results(st, cmd, "spxtregress", &p, &sp, &names, &b, &v, &scal, &macros);
        Ok(out.join("\n"))
    }
}

/// Layout for panel models: internal order [x..., λ?, ρ?, extra...].
fn xt_layout(xnames: &[String], sp: &Spec, has_l: bool, has_r: bool, extra: &[String]) -> (Vec<String>, Vec<usize>) {
    let y = &sp.depvar;
    let k = xnames.len();
    let mut names = Vec::new();
    let mut idx = Vec::new();
    let mut lag_pos: Vec<(String, String, usize)> = Vec::new();
    for (j, nm) in xnames.iter().enumerate() {
        if let Some((w, v)) = nm.split_once(':') {
            lag_pos.push((w.to_string(), v.to_string(), j));
        } else if nm != "_cons" {
            names.push(format!("{y}:{nm}"));
            idx.push(j);
        }
    }
    if let Some(j) = xnames.iter().position(|n| n == "_cons") {
        names.push(format!("{y}:_cons"));
        idx.push(j);
    }
    let mut mats: Vec<String> = Vec::new();
    for m in sp.dl.iter().chain(lag_pos.iter().map(|l| &l.0)).chain(sp.el.iter()) {
        if !mats.contains(m) {
            mats.push(m.clone());
        }
    }
    for m in &mats {
        if has_l && sp.dl.as_deref() == Some(m.as_str()) {
            names.push(format!("{m}:{y}"));
            idx.push(k);
        }
        for l in &lag_pos {
            if &l.0 == m {
                names.push(format!("{m}:{}", l.1));
                idx.push(l.2);
            }
        }
        if has_r && sp.el.as_deref() == Some(m.as_str()) {
            names.push(format!("{m}:e.{y}"));
            idx.push(k + has_l as usize);
        }
    }
    let base = k + has_l as usize + has_r as usize;
    for (j, e) in extra.iter().enumerate() {
        names.push(e.clone());
        idx.push(base + j);
    }
    (names, idx)
}

// ---------------------------------------------------------------------------
// Postestimation
// ---------------------------------------------------------------------------

struct Fitted {
    depvar: String,
    /// data rows (period-major), units per period
    rows: Vec<usize>,
    n: usize,
    w: Option<Sparse>,
    lam: f64,
    /// X β + (W X) θ over `rows`
    xb: Vec<f64>,
    y: Vec<f64>,
    names: Vec<String>,
    b: Vec<f64>,
    v: Matrix,
}

fn fitted(ds: &DataSet, st: &Store) -> Result<Fitted, String> {
    let e = &st.e;
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?.clone();
    let v = e.matrices.get("V").cloned().unwrap_or_else(|| Matrix::zeros(b.cols, b.cols));
    let depvar = e.macros.get("depvar").cloned().unwrap_or_default();
    let dl = e.macros.get("dlmat").cloned().unwrap_or_default();
    let sample = e.sample.clone().ok_or_else(|| rc_err(301, "estimation sample not found"))?;
    let s = need_spset(ds)?;
    let idv = util::col(ds, &s.id)?;
    let panel = e.macros.get("cmd").is_some_and(|c| c == "spxtregress");
    let timev = if panel { panel_vars(st).map(|(_, t)| util::col(ds, &t)).transpose()? } else { None };
    let mut rows: Vec<usize> = (0..ds.n_obs.min(sample.len())).filter(|&i| sample[i]).collect();
    rows.sort_by(|&a, &b| {
        let ta = timev.as_ref().map(|t| t[a]).unwrap_or(0.0);
        let tb = timev.as_ref().map(|t| t[b]).unwrap_or(0.0);
        ta.total_cmp(&tb).then(idv[a].total_cmp(&idv[b]))
    });
    let nt = rows.len();
    let t = match &timev {
        Some(tv) => {
            let mut p: Vec<f64> = rows.iter().map(|&r| tv[r]).collect();
            p.dedup();
            p.len().max(1)
        }
        None => 1,
    };
    let n = nt / t;
    let ids: Vec<f64> = rows[..n].iter().map(|&r| idv[r]).collect();
    let mut mats: HashMap<String, Sparse> = HashMap::new();
    let mut get = |name: &str| -> Result<Sparse, String> {
        if let Some(m) = mats.get(name) {
            return Ok(m.clone());
        }
        let m = Sparse::from_dense(&align(&get_spmat(st, name)?, &ids, name, true)?);
        mats.insert(name.to_string(), m.clone());
        Ok(m)
    };
    let env = Env::with_store(ds, st);
    let mut xb = vec![0.0; nt];
    let mut lam = 0.0;
    let names = b.colnames.clone();
    for (j, nm) in names.iter().enumerate() {
        let coef = b.data[j];
        let (eq, var) = nm.split_once(':').unwrap_or(("", nm));
        if eq == "/" || eq.starts_with("var(") {
            continue;
        }
        if eq == depvar || eq.is_empty() {
            if coef == 0.0 {
                continue;
            }
            for (k, &r) in rows.iter().enumerate() {
                let x = crate::postest::coef_value(&env, var, r)?;
                xb[k] += coef * x;
            }
        } else if var == depvar && eq == dl {
            lam = coef;
        } else if var.starts_with("e.") {
            continue;
        } else {
            let w = get(eq)?;
            let x = util::col(ds, var)?;
            let xs: Vec<f64> = rows.iter().map(|&r| x[r]).collect();
            let wx = w.mv_blocks(&xs);
            for k in 0..nt {
                xb[k] += coef * wx[k];
            }
        }
    }
    let w = if dl.is_empty() { None } else { Some(get(&dl)?) };
    let yv = util::col(ds, &depvar)?;
    let y = rows.iter().map(|&r| yv[r]).collect();
    Ok(Fitted { depvar, rows, n, w, lam, xb, y, names, b: b.data.clone(), v })
}

/// predict after spregress / spivregress / spxtregress.
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<Option<String>, String> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    if !matches!(ecmd.as_str(), "spregress" | "spivregress" | "spxtregress") {
        return Ok(None);
    }
    let words: Vec<&str> = cmd.main.split_whitespace().collect();
    let (ty, name) = match words.as_slice() {
        [t, v] => (VarType::from_type_str(t).ok_or_else(|| rc_err(198, format!("{t} invalid type")))?, v.to_string()),
        [v] => (VarType::Float, v.to_string()),
        _ => return Err(rc_err(198, "predict newvar [, statistic]")),
    };
    crate::varlist::check_name(&name)?;
    if ds.var_index.contains_key(&name) {
        return Err(rc_err(110, format!("variable {name} already defined")));
    }
    let f = fitted(ds, st)?;
    let stat = ["rform", "xb", "naive", "direct", "indirect", "residuals", "ucresidual"]
        .iter()
        .find(|s| cmd.opts.iter().any(|(o, _)| s.starts_with(o.to_lowercase().as_str()) && o.len() >= 2))
        .copied()
        .unwrap_or("rform");
    let nt = f.rows.len();
    let n = f.n;
    let blocks = |v: &[f64], g: &dyn Fn(&[f64]) -> Vec<f64>| -> Vec<f64> { v.chunks(n).flat_map(|c| g(c)).collect() };
    let vals: Vec<f64> = match stat {
        "xb" => f.xb.clone(),
        "rform" => match &f.w {
            Some(w) => blocks(&f.xb, &|c| solve_ilw(w, f.lam, c)),
            None => f.xb.clone(),
        },
        "naive" => match &f.w {
            Some(w) => {
                let wy = w.mv_blocks(&f.y);
                (0..nt).map(|k| f.xb[k] + f.lam * wy[k]).collect()
            }
            None => f.xb.clone(),
        },
        "residuals" | "ucresidual" => {
            let wy = f.w.as_ref().map(|w| w.mv_blocks(&f.y)).unwrap_or_else(|| vec![0.0; nt]);
            (0..nt).map(|k| f.y[k] - f.lam * wy[k] - f.xb[k]).collect()
        }
        "direct" | "indirect" => {
            let s = match &f.w {
                Some(w) => ilw_inverse(w, f.lam)?,
                None => Matrix::identity(n),
            };
            let mut outv = Vec::with_capacity(nt);
            for c in f.xb.chunks(n) {
                let tot = mat_vec(&s, c);
                for i in 0..n {
                    let d = s[(i, i)] * c[i];
                    outv.push(if stat == "direct" { d } else { tot[i] - d });
                }
            }
            outv
        }
        _ => unreachable!(),
    };
    let mut data = vec![SYSMISS; ds.n_obs];
    for (k, &r) in f.rows.iter().enumerate() {
        data[r] = vals[k];
    }
    let mut v = Variable::new(&name, ty);
    v.data = data.iter().map(|&x| crate::data::store_num(ty, x)).collect();
    ds.add_variable(v)?;
    let _ = &f.depvar;
    Ok(Some(String::new()))
}

/// estat subcommands handled here: `impact` after the spatial models and
/// `moran` after regress.
pub fn estat(ds: &DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    match sub {
        "impact" | "impacts" if matches!(ecmd.as_str(), "spregress" | "spivregress" | "spxtregress") => Some(impact(ds, st, cmd)),
        "moran" => Some(moran(ds, st, cmd)),
        _ => None,
    }
}

fn impact(ds: &DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let f = fitted(ds, st)?;
    let n = f.n;
    let nf = n as f64;
    let dl = st.e.macros.get("dlmat").cloned().unwrap_or_default();
    let level = cmd.opt("level", 1).and_then(|s| s.trim().parse().ok()).unwrap_or(st.level);
    // covariates: depvar-equation variables and spatially lagged covariates
    let mut vars: Vec<String> = Vec::new();
    for nm in &f.names {
        let (eq, var) = nm.split_once(':').unwrap_or(("", nm));
        if eq == "/" || eq.starts_with("var(") || var == "_cons" || var.starts_with("e.") || var == f.depvar {
            continue;
        }
        if !vars.iter().any(|v| v == var) {
            vars.push(var.to_string());
        }
    }
    let want: Vec<String> = cmd.main.split_whitespace().skip(1).map(String::from).collect();
    if !want.is_empty() {
        let expanded: Vec<String> = want.iter().flat_map(|w| crate::varlist::expand(ds, w).unwrap_or_else(|_| vec![w.clone()])).collect();
        vars.retain(|v| expanded.contains(v));
        if vars.is_empty() {
            return Err(rc_err(111, "no covariates to compute impacts for"));
        }
    }
    let pos = |nm: &str| f.names.iter().position(|x| x == nm);
    let lam_i = pos(&format!("{dl}:{}", f.depvar));
    // scalars of S = (I − λW)^{-1}
    let (a, bq, c, d, da, dbq, dc, dd) = match &f.w {
        Some(w) => {
            let s = ilw_inverse(w, f.lam)?;
            // SW
            let mut sw = Matrix::zeros(n, n);
            for i in 0..n {
                for (k, r) in w.rows.iter().enumerate() {
                    let sik = s[(i, k)];
                    if sik == 0.0 {
                        continue;
                    }
                    for &(j, v) in r {
                        sw[(i, j)] += sik * v;
                    }
                }
            }
            let ones = vec![1.0; n];
            let s1 = mat_vec(&s, &ones);
            let sw1 = mat_vec(&sw, &ones);
            let one_sw: Vec<f64> = (0..n).map(|j| (0..n).map(|i| sw[(i, j)]).sum()).collect();
            let tr_s = s.trace();
            let tr_sw = sw.trace();
            let mut tr_sws = 0.0;
            let mut tr_swsw = 0.0;
            for i in 0..n {
                for j in 0..n {
                    tr_sws += sw[(i, j)] * s[(j, i)];
                    tr_swsw += sw[(i, j)] * sw[(j, i)];
                }
            }
            (tr_s / nf, tr_sw / nf, s1.iter().sum::<f64>() / nf, sw1.iter().sum::<f64>() / nf, tr_sws / nf, tr_swsw / nf, dot(&one_sw, &s1) / nf, dot(&one_sw, &sw1) / nf)
        }
        None => (1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    };
    // a = tr(S)/n, bq = tr(SW)/n, c = 1'S1/n, d = 1'SW1/n and their λ-derivatives
    let kp = f.b.len();
    let mut rows_out: Vec<(String, Vec<f64>, Vec<f64>)> = Vec::new(); // (eq:var, value?, gradient)
    let mut bvals = [Vec::new(), Vec::new(), Vec::new()];
    let mut grads: [Vec<Vec<f64>>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for v in &vars {
        let bi = pos(&format!("{}:{v}", f.depvar));
        // spatial lags of this variable under the dvarlag matrix
        let ti = if dl.is_empty() { None } else { pos(&format!("{dl}:{v}")) };
        let beta = bi.map(|i| f.b[i]).unwrap_or(0.0);
        let theta = ti.map(|i| f.b[i]).unwrap_or(0.0);
        let direct = beta * a + theta * bq;
        let total = beta * c + theta * d;
        let mut gd = vec![0.0; kp];
        let mut gt = vec![0.0; kp];
        if let Some(i) = bi {
            gd[i] = a;
            gt[i] = c;
        }
        if let Some(i) = ti {
            gd[i] = bq;
            gt[i] = d;
        }
        if let Some(i) = lam_i {
            gd[i] = beta * da + theta * dbq;
            gt[i] = beta * dc + theta * dd;
        }
        let gi: Vec<f64> = gt.iter().zip(&gd).map(|(x, y)| x - y).collect();
        bvals[0].push(direct);
        bvals[1].push(total - direct);
        bvals[2].push(total);
        grads[0].push(gd);
        grads[1].push(gi);
        grads[2].push(gt);
    }
    let _ = &mut rows_out;
    let kv = vars.len();
    let mut names = Vec::new();
    let mut b = Vec::new();
    let mut jac: Vec<Vec<f64>> = Vec::new();
    for (e, lab) in ["direct", "indirect", "total"].iter().enumerate() {
        for k in 0..kv {
            names.push(format!("{lab}:{}", vars[k]));
            b.push(bvals[e][k]);
            jac.push(grads[e][k].clone());
        }
    }
    let m = names.len();
    let mut vv = Matrix::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            let mut s = 0.0;
            for p in 0..kp {
                if jac[i][p] == 0.0 {
                    continue;
                }
                for q in 0..kp {
                    s += jac[i][p] * f.v[(p, q)] * jac[j][q];
                }
            }
            vv[(i, j)] = s;
        }
    }
    let mut out = vec!["progress   :100%".to_string(), String::new(), head_line("Average impacts", "Number of obs", &util::commas(f.rows.len())), String::new()];
    let lines = est::coef_table(&est::TableSpec { depvar: "", names: &names, b: &b, v: &vv, omitted: &vec![false; m], is_base: &vec![false; m], df: None, level, eform: None, coef_title: "dy/dx" });
    for (k, l) in lines.into_iter().enumerate() {
        if k == 1 {
            let lw = l.find('|').unwrap_or(13);
            out.push(format!("{:lw$}|            Delta-Method", ""));
            out.push(l.replace("Std. err.", "std. err."));
        } else {
            out.push(l);
        }
    }
    let tbl = est::take_table();
    let r = st.begin_r();
    if let Some((m, lvl)) = tbl {
        r.set_mat("table", m);
        r.set("level", lvl);
    }
    r.set("N", f.rows.len() as f64);
    for (e, lab) in ["direct", "indirect", "total"].iter().enumerate() {
        let bm = Matrix::row_vector(&bvals[e]).with_names(vec!["r1".into()], vars.clone());
        let mut vm = Matrix::zeros(kv, kv);
        for i in 0..kv {
            for j in 0..kv {
                vm[(i, j)] = vv[(e * kv + i, e * kv + j)];
            }
        }
        r.set_mat(&format!("b_{lab}"), bm);
        r.set_mat(&format!("V_{lab}"), vm.with_names(vars.clone(), vars.clone()));
    }
    Ok(out.join("\n"))
}

fn moran(ds: &DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    if ecmd != "regress" {
        return Err(rc_err(321, "estat moran is allowed only after regress"));
    }
    let mats: Vec<String> = cmd.opts.iter().filter(|(n, _)| n.to_lowercase().starts_with("errorl")).filter_map(|(_, a)| a.clone()).flat_map(|a| a.split_whitespace().map(String::from).collect::<Vec<_>>()).collect();
    if mats.is_empty() {
        return Err(rc_err(198, "option errorlag() required"));
    }
    let s = need_spset(ds)?;
    let b = st.e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let depvar = st.e.macros.get("depvar").cloned().unwrap_or_default();
    let sample = st.e.sample.clone().ok_or_else(|| rc_err(301, "estimation sample not found"))?;
    let idv = util::col(ds, &s.id)?;
    let mut rows: Vec<usize> = (0..ds.n_obs.min(sample.len())).filter(|&i| sample[i]).collect();
    rows.sort_by(|&a, &c| idv[a].total_cmp(&idv[c]));
    let ids: Vec<f64> = rows.iter().map(|&r| idv[r]).collect();
    let env = Env::with_store(ds, st);
    let yv = util::col(ds, &depvar)?;
    let mut e = Vec::with_capacity(rows.len());
    for &r in &rows {
        e.push(yv[r] - crate::postest::xb_row(&env, &b, None, r)?);
    }
    let n = e.len() as f64;
    let s2 = dot(&e, &e) / n;
    let ws: Vec<Sparse> = mats.iter().map(|m| Ok(Sparse::from_dense(&align(&get_spmat(st, m)?, &ids, m, cmd.has_opt("force", 5))?))).collect::<Result<_, String>>()?;
    let k = ws.len();
    let q: Vec<f64> = ws.iter().map(|w| w.quad(&e, &e) / s2).collect();
    let mut phi = Matrix::zeros(k, k);
    for a in 0..k {
        for c in 0..k {
            // tr(A_a A_c' + A_a A_c) = Σ A_a∘A_c + Σ A_a∘A_c'
            phi[(a, c)] = weighted_frob(&ws[a], &ws[c], None) + weighted_frob(&ws[a], &ws[c].t(), None);
        }
    }
    let chi2 = dot(&q, &mat_vec(&inv(&phi), &q));
    let p = dist::chi2tail(k as f64, chi2);
    let r = st.begin_r();
    r.set("chi2", chi2);
    r.set("df", k as f64);
    r.set("p", p);
    Ok([
        "Moran test for spatial dependence".to_string(),
        "         H0: Error terms are i.i.d.".into(),
        format!("         Errorlags:  {}", mats.join(" ")),
        String::new(),
        format!("         chi2({k})      = {:>8.2}", chi2),
        format!("         Prob > chi2  = {:>8.4}", p),
    ]
    .join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_ops() {
        let mut m = Matrix::zeros(3, 3);
        m[(0, 1)] = 1.0;
        m[(1, 2)] = 2.0;
        m[(2, 0)] = 3.0;
        let s = Sparse::from_dense(&m);
        let p = s.mul(&s);
        assert_eq!(sparse_dense(&p)[(0, 2)], 2.0);
        assert_eq!(s.t().mv(&[1.0, 1.0, 1.0]), vec![3.0, 1.0, 2.0]);
        let x = solve_ilw(&s, 0.1, &[1.0, 2.0, 3.0]);
        let back: Vec<f64> = x.iter().zip(s.mv(&x)).map(|(a, b)| a - 0.1 * b).collect();
        for (a, b) in back.iter().zip([1.0, 2.0, 3.0]) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
