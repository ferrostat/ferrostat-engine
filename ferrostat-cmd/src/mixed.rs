//! `mixed`: linear mixed-effects models with nested random effects, fitted
//! by maximum likelihood or REML. The fixed effects and the residual
//! variance are profiled out; the relative random-effects covariance
//! parameters are optimised by Newton-Raphson, and standard errors come
//! from the Hessian of the unprofiled likelihood in the log-standard
//! deviation / atanh-correlation metric.

use std::collections::HashMap;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value};
use ferrostat_parser::Command;

use crate::choice_mixed::{f2, f4, hrow, zcrit};
use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, DesignSpec, Likelihood};
use crate::eval::{rc_err, Env};
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Cov {
    Ind,
    Ident,
    Exch,
    Uns,
}

impl Cov {
    fn parse(s: &str) -> Result<Cov, String> {
        let s = s.trim().to_lowercase();
        if s.is_empty() {
            return Ok(Cov::Ind);
        }
        for (name, min, c) in [("independent", 3, Cov::Ind), ("identity", 2, Cov::Ident), ("exchangeable", 2, Cov::Exch), ("unstructured", 2, Cov::Uns)] {
            if s.len() >= min && name.starts_with(s.as_str()) {
                return Ok(c);
            }
        }
        Err(rc_err(198, format!("covariance({s}) not allowed")))
    }
    #[allow(dead_code)]
    fn label(self) -> &'static str {
        match self {
            Cov::Ind => "Independent",
            Cov::Ident => "Identity",
            Cov::Exch => "Exchangeable",
            Cov::Uns => "Unstructured",
        }
    }
    fn key(self) -> &'static str {
        match self {
            Cov::Ind => "independent",
            Cov::Ident => "identity",
            Cov::Exch => "exchangeable",
            Cov::Uns => "unstructured",
        }
    }
    fn npar(self, q: usize) -> usize {
        match self {
            Cov::Ind => q,
            Cov::Ident => 1,
            Cov::Exch => {
                if q > 1 { 2 } else { 1 }
            }
            Cov::Uns => q * (q + 1) / 2,
        }
    }
    /// Which parameters are log standard deviations (shifted by ln sigma_e
    /// when moving between relative and absolute scales).
    fn is_lnsd(self, q: usize) -> Vec<bool> {
        match self {
            Cov::Ind => vec![true; q],
            Cov::Ident => vec![true],
            Cov::Exch => {
                if q > 1 { vec![true, false] } else { vec![true] }
            }
            Cov::Uns => (0..q * (q + 1) / 2).map(|i| i < q).collect(),
        }
    }
    /// Covariance matrix of the random effects from (log sd, atanh corr).
    fn psi(self, q: usize, th: &[f64], shift: f64) -> Matrix {
        let mut m = Matrix::zeros(q, q);
        match self {
            Cov::Ind => {
                for r in 0..q {
                    m[(r, r)] = (2.0 * (th[r] + shift)).exp();
                }
            }
            Cov::Ident => {
                for r in 0..q {
                    m[(r, r)] = (2.0 * (th[0] + shift)).exp();
                }
            }
            Cov::Exch => {
                let v = (2.0 * (th[0] + shift)).exp();
                let c = if q > 1 { th[1].tanh() } else { 0.0 };
                for r in 0..q {
                    for s in 0..q {
                        m[(r, s)] = if r == s { v } else { c * v };
                    }
                }
            }
            Cov::Uns => {
                let sd: Vec<f64> = (0..q).map(|r| (th[r] + shift).exp()).collect();
                let mut k = q;
                for r in 0..q {
                    m[(r, r)] = sd[r] * sd[r];
                    for s in r + 1..q {
                        let c = th[k].tanh() * sd[r] * sd[s];
                        m[(r, s)] = c;
                        m[(s, r)] = c;
                        k += 1;
                    }
                }
            }
        }
        m
    }
}

/// Random-effects structure over the estimation sample.
struct Structure {
    /// per level: random-effect design (q columns over the sample)
    z: Vec<Vec<Vec<f64>>>,
    /// per level: unit index of each observation (nested within levels above)
    ids: Vec<Vec<usize>>,
    /// rows of each top-level group
    tops: Vec<Vec<usize>>,
    /// per level: size of each unit
    sizes: Vec<Vec<usize>>,
}

fn codes_of(ds: &DataSet, var: &str, rows: &[usize]) -> Result<Vec<String>, String> {
    let v = ds.get_var(var).ok_or_else(|| rc_err(111, format!("variable {var} not found")))?;
    Ok(rows
        .iter()
        .map(|&i| match &v.data[i] {
            Value::Str(s) => s.clone(),
            x => fmt_g(x.num(), 17),
        })
        .collect())
}

/// Build the nested structure: `revars[l]` are the random-effect variables
/// at level `l` (`_cons` for the intercept).
fn structure(ds: &DataSet, rows: &[usize], ivars: &[String], revars: &[Vec<String>]) -> Result<Structure, String> {
    let n = rows.len();
    let mut z = Vec::new();
    for vs in revars {
        let mut cols = Vec::new();
        for v in vs {
            if v == "_cons" {
                cols.push(vec![1.0; n]);
            } else {
                let c = util::col(ds, v)?;
                cols.push(rows.iter().map(|&i| c[i]).collect());
            }
        }
        z.push(cols);
    }
    let mut ids = Vec::new();
    let mut sizes = Vec::new();
    let mut prefix = vec![String::new(); n];
    for iv in ivars {
        let c = codes_of(ds, iv, rows)?;
        let mut map: HashMap<String, usize> = HashMap::new();
        let mut id = vec![0usize; n];
        let mut sz: Vec<usize> = Vec::new();
        // order units by their (nested) key in sorted order of first appearance
        let mut keys: Vec<(String, usize)> = Vec::new();
        for t in 0..n {
            prefix[t] = format!("{}\u{1}{}", prefix[t], c[t]);
            if !map.contains_key(&prefix[t]) {
                map.insert(prefix[t].clone(), keys.len());
                keys.push((prefix[t].clone(), 0));
                sz.push(0);
            }
            let u = map[&prefix[t]];
            id[t] = u;
            sz[u] += 1;
        }
        ids.push(id);
        sizes.push(sz);
    }
    let ntop = sizes.first().map_or(0, |s| s.len());
    let mut tops = vec![Vec::new(); ntop];
    for t in 0..n {
        tops[ids[0][t]].push(t);
    }
    Ok(Structure { z, ids, tops, sizes })
}

struct Acc {
    logdet: f64,
    xvx: Matrix,
    xvy: Vec<f64>,
    yvy: f64,
}

/// In-place Cholesky of a row-major n x n matrix (lower triangle).
fn chol(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut s = a[j * n + j];
        for k in 0..j {
            s -= a[j * n + k] * a[j * n + k];
        }
        if !(s > 0.0) || !s.is_finite() {
            return false;
        }
        let d = s.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / d;
        }
    }
    true
}

fn fsolve(l: &[f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
}

fn bsolve(l: &[f64], n: usize, b: &mut [f64]) {
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= l[k * n + i] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
}

/// Cross products of a top-level group with its random-effects design
/// Z (all units of all levels inside the group, Q columns).
struct GroupPre {
    /// (level, offset) of each random-effects block
    blocks: Vec<(usize, usize)>,
    /// column block of each observation per level
    nq: usize,
    ztz: Vec<f64>,
    ztx: Vec<f64>,
    zty: Vec<f64>,
    xtx: Vec<f64>,
    xty: Vec<f64>,
    yty: f64,
    n: usize,
}

struct Mx<'a> {
    y: &'a [f64],
    x: Vec<&'a [f64]>,
    st: &'a Structure,
    covs: Vec<Cov>,
    reml: bool,
    pre: Vec<GroupPre>,
}

/// Lower Cholesky-type factor of a positive semi-definite matrix.
fn psd_factor(m: &Matrix) -> Matrix {
    match m.cholesky() {
        Ok(l) => l,
        Err(_) => {
            let (vals, vecs) = m.sym_eigen();
            let q = m.rows;
            let mut l = Matrix::zeros(q, q);
            for i in 0..q {
                for k in 0..q {
                    l[(i, k)] = vecs[(i, k)] * vals[k].max(0.0).sqrt();
                }
            }
            l
        }
    }
}

/// Nonzero random-effects design entries (column, value) of observation t
/// within its top-level group, given the block offsets per (level, unit).
fn zrow(st: &Structure, t: usize, offs: &HashMap<(usize, usize), usize>) -> Vec<(usize, f64)> {
    let mut v = Vec::new();
    for l in 0..st.z.len() {
        let o = offs[&(l, st.ids[l][t])];
        for (r, zc) in st.z[l].iter().enumerate() {
            if zc[t] != 0.0 {
                v.push((o + r, zc[t]));
            }
        }
    }
    v
}

/// Block offsets of the units of each level inside a top-level group.
fn group_offsets(st: &Structure, rows: &[usize]) -> (HashMap<(usize, usize), usize>, Vec<(usize, usize)>, usize) {
    let mut offs = HashMap::new();
    let mut blocks = Vec::new();
    let mut nq = 0;
    for &t in rows {
        for l in 0..st.z.len() {
            let key = (l, st.ids[l][t]);
            if let std::collections::hash_map::Entry::Vacant(e) = offs.entry(key) {
                e.insert(nq);
                blocks.push((l, nq));
                nq += st.z[l].len();
            }
        }
    }
    (offs, blocks, nq)
}

impl<'a> Mx<'a> {
    fn new(y: &'a [f64], x: Vec<&'a [f64]>, st: &'a Structure, covs: Vec<Cov>, reml: bool) -> Self {
        let p = x.len();
        let mut pre = Vec::new();
        for rows in &st.tops {
            let (offs, blocks, nq) = group_offsets(st, rows);
            let mut g = GroupPre { blocks, nq, ztz: vec![0.0; nq * nq], ztx: vec![0.0; nq * p], zty: vec![0.0; nq], xtx: vec![0.0; p * p], xty: vec![0.0; p], yty: 0.0, n: rows.len() };
            for &t in rows {
                let zr = zrow(st, t, &offs);
                let yt = y[t];
                for &(a, va) in &zr {
                    for &(b, vb) in &zr {
                        g.ztz[a * nq + b] += va * vb;
                    }
                    for j in 0..p {
                        g.ztx[a * p + j] += va * x[j][t];
                    }
                    g.zty[a] += va * yt;
                }
                for i in 0..p {
                    for j in 0..p {
                        g.xtx[i * p + j] += x[i][t] * x[j][t];
                    }
                    g.xty[i] += x[i][t] * yt;
                }
                g.yty += yt * yt;
            }
            pre.push(g);
        }
        Mx { y, x, st, covs, reml, pre }
    }
    fn n(&self) -> usize {
        self.y.len()
    }
    fn p(&self) -> usize {
        self.x.len()
    }
    fn q(&self, l: usize) -> usize {
        self.st.z[l].len()
    }
    fn ntheta(&self) -> usize {
        (0..self.covs.len()).map(|l| self.covs[l].npar(self.q(l))).sum()
    }
    fn psis(&self, th: &[f64], shift: f64) -> Vec<Matrix> {
        let mut out = Vec::new();
        let mut o = 0;
        for (l, c) in self.covs.iter().enumerate() {
            let q = self.q(l);
            let k = c.npar(q);
            out.push(c.psi(q, &th[o..o + k], shift));
            o += k;
        }
        out
    }
    /// L' A for the block-diagonal factor L (A has `cols` columns, row-major).
    fn lt_mul(g: &GroupPre, lf: &[Matrix], a: &[f64], cols: usize) -> Vec<f64> {
        let mut out = vec![0.0; g.nq * cols];
        for &(l, o) in &g.blocks {
            let f = &lf[l];
            let q = f.rows;
            for r in 0..q {
                for k in r..q {
                    let fv = f[(k, r)];
                    if fv == 0.0 {
                        continue;
                    }
                    for c in 0..cols {
                        out[(o + r) * cols + c] += fv * a[(o + k) * cols + c];
                    }
                }
            }
        }
        out
    }
    /// L v for the block-diagonal factor L.
    fn l_mul(g: &GroupPre, lf: &[Matrix], v: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; g.nq];
        for &(l, o) in &g.blocks {
            let f = &lf[l];
            let q = f.rows;
            for r in 0..q {
                for k in 0..=r {
                    out[o + r] += f[(r, k)] * v[o + k];
                }
            }
        }
        out
    }
    /// Cholesky factor of M = I + L'Z'ZL / s2 for a group.
    fn mfactor(g: &GroupPre, lf: &[Matrix], s2: f64) -> Option<Vec<f64>> {
        let nq = g.nq;
        // L' (Z'Z) L = L' (L' (Z'Z))'  (Z'Z symmetric)
        let a1 = Self::lt_mul(g, lf, &g.ztz, nq);
        let mut a1t = vec![0.0; nq * nq];
        for i in 0..nq {
            for j in 0..nq {
                a1t[j * nq + i] = a1[i * nq + j];
            }
        }
        let a = Self::lt_mul(g, lf, &a1t, nq);
        let mut m = vec![0.0; nq * nq];
        for i in 0..nq {
            for j in 0..nq {
                m[i * nq + j] = 0.5 * (a[i * nq + j] + a[j * nq + i]) / s2 + if i == j { 1.0 } else { 0.0 };
            }
        }
        if chol(&mut m, nq) { Some(m) } else { None }
    }
    fn acc(&self, psis: &[Matrix], s2: f64) -> Option<Acc> {
        let p = self.p();
        let lf: Vec<Matrix> = psis.iter().map(psd_factor).collect();
        let mut acc = Acc { logdet: 0.0, xvx: Matrix::zeros(p, p), xvy: vec![0.0; p], yvy: 0.0 };
        if !(s2 > 0.0) || !s2.is_finite() {
            return None;
        }
        for g in &self.pre {
            let nq = g.nq;
            let m = Self::mfactor(g, &lf, s2)?;
            let mut ld = g.n as f64 * s2.ln();
            for i in 0..nq {
                ld += 2.0 * m[i * nq + i].ln();
            }
            acc.logdet += ld;
            // W = L' [Z'X | Z'y], then M^-1/2 W via the forward solve
            let mut zxy = vec![0.0; nq * (p + 1)];
            for i in 0..nq {
                for j in 0..p {
                    zxy[i * (p + 1) + j] = g.ztx[i * p + j];
                }
                zxy[i * (p + 1) + p] = g.zty[i];
            }
            let w = Self::lt_mul(g, &lf, &zxy, p + 1);
            let mut cols: Vec<Vec<f64>> = (0..=p).map(|c| (0..nq).map(|i| w[i * (p + 1) + c]).collect()).collect();
            for c in cols.iter_mut() {
                fsolve(&m, nq, c);
            }
            let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(u, v)| u * v).sum::<f64>();
            for i in 0..p {
                for j in 0..p {
                    acc.xvx[(i, j)] += (g.xtx[i * p + j] - dot(&cols[i], &cols[j]) / s2) / s2;
                }
                acc.xvy[i] += (g.xty[i] - dot(&cols[i], &cols[p]) / s2) / s2;
            }
            acc.yvy += (g.yty - dot(&cols[p], &cols[p]) / s2) / s2;
        }
        Some(acc)
    }
    /// GLS estimate, residual quadratic form and ln|X'V^-1 X|.
    fn gls(acc: &Acc) -> Option<(Vec<f64>, f64, f64)> {
        let p = acc.xvy.len();
        if p == 0 {
            return Some((vec![], acc.yvy, 0.0));
        }
        let l = acc.xvx.cholesky().ok()?;
        let ld: f64 = (0..p).map(|i| 2.0 * l[(i, i)].ln()).sum();
        let (inv, _) = acc.xvx.sym_inverse(1e-14);
        let b: Vec<f64> = (0..p).map(|i| (0..p).map(|j| inv[(i, j)] * acc.xvy[j]).sum()).collect();
        let rss = acc.yvy - b.iter().zip(&acc.xvy).map(|(a, c)| a * c).sum::<f64>();
        Some((b, rss, ld))
    }
    /// Profiled log (restricted) likelihood in the relative parameters.
    fn profiled(&self, th: &[f64]) -> f64 {
        let psis = self.psis(th, 0.0);
        let Some(acc) = self.acc(&psis, 1.0) else { return f64::NEG_INFINITY };
        let Some((_, rss, ld)) = Self::gls(&acc) else { return f64::NEG_INFINITY };
        if !(rss > 0.0) {
            return f64::NEG_INFINITY;
        }
        let n = self.n() as f64;
        let l2p = (2.0 * std::f64::consts::PI).ln();
        if self.reml {
            let d = n - self.p() as f64;
            -0.5 * (d * (1.0 + l2p + (rss / d).ln()) + acc.logdet + ld)
        } else {
            -0.5 * (n * (1.0 + l2p + (rss / n).ln()) + acc.logdet)
        }
    }
    /// sigma_e^2 at the profiled optimum.
    fn sigma2(&self, th: &[f64]) -> f64 {
        let psis = self.psis(th, 0.0);
        let acc = self.acc(&psis, 1.0).unwrap();
        let (_, rss, _) = Self::gls(&acc).unwrap();
        let d = if self.reml { self.n() - self.p() } else { self.n() };
        rss / d as f64
    }
    /// Unprofiled log likelihood in (beta, absolute parameters, ln sigma_e).
    fn full_ml(&self, beta: &[f64], phi: &[f64]) -> f64 {
        let m = phi.len();
        let psis = self.psis(&phi[..m - 1], 0.0);
        let s2 = (2.0 * phi[m - 1]).exp();
        let Some(acc) = self.acc(&psis, s2) else { return f64::NEG_INFINITY };
        let p = self.p();
        let mut qf = acc.yvy;
        for a in 0..p {
            qf -= 2.0 * beta[a] * acc.xvy[a];
            for b in 0..p {
                qf += beta[a] * acc.xvx[(a, b)] * beta[b];
            }
        }
        -0.5 * (self.n() as f64 * (2.0 * std::f64::consts::PI).ln() + acc.logdet + qf)
    }
    /// Restricted log likelihood in the absolute parameters.
    fn full_reml(&self, phi: &[f64]) -> f64 {
        let m = phi.len();
        let psis = self.psis(&phi[..m - 1], 0.0);
        let s2 = (2.0 * phi[m - 1]).exp();
        let Some(acc) = self.acc(&psis, s2) else { return f64::NEG_INFINITY };
        let Some((_, rss, ld)) = Self::gls(&acc) else { return f64::NEG_INFINITY };
        let d = (self.n() - self.p()) as f64;
        -0.5 * (d * (2.0 * std::f64::consts::PI).ln() + acc.logdet + ld + rss)
    }
}

/// A likelihood with numerical derivatives.
struct NumLik<'a> {
    f: &'a dyn Fn(&[f64]) -> f64,
    k: usize,
}

fn num_grad_hess(f: &dyn Fn(&[f64]) -> f64, x: &[f64], h: f64) -> (f64, Vec<f64>, Matrix) {
    let k = x.len();
    let f0 = f(x);
    let mut g = vec![0.0; k];
    let mut hm = Matrix::zeros(k, k);
    let at = |d: &[(usize, f64)]| {
        let mut y = x.to_vec();
        for &(i, s) in d {
            y[i] += s;
        }
        f(&y)
    };
    for i in 0..k {
        let (fp, fm) = (at(&[(i, h)]), at(&[(i, -h)]));
        g[i] = (fp - fm) / (2.0 * h);
        hm[(i, i)] = (fp - 2.0 * f0 + fm) / (h * h);
        for j in 0..i {
            let v = (at(&[(i, h), (j, h)]) - at(&[(i, h), (j, -h)]) - at(&[(i, -h), (j, h)]) + at(&[(i, -h), (j, -h)])) / (4.0 * h * h);
            hm[(i, j)] = v;
            hm[(j, i)] = v;
        }
    }
    (f0, g, hm)
}

impl<'a> Likelihood for NumLik<'a> {
    fn k(&self) -> usize {
        self.k
    }
    fn eval(&self, b: &[f64], _need: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        let f0 = (self.f)(b);
        if !f0.is_finite() {
            return (f64::NEG_INFINITY, vec![0.0; self.k], Matrix::identity(self.k), vec![]);
        }
        let (f0, g, h) = num_grad_hess(self.f, b, 1e-4);
        (f0, g, h.scale(-1.0), vec![])
    }
}

/// Names of the random-effects parameters of a level in the estimation
/// metric (as in e(b)) and for display.
fn re_names(l: usize, c: Cov, zn: &[String]) -> (Vec<String>, Vec<String>) {
    let q = zn.len();
    let mut est = Vec::new();
    let mut disp = Vec::new();
    match c {
        Cov::Ind => {
            for r in 0..q {
                est.push(format!("lns{}_1_{}:_cons", l + 1, r + 1));
                disp.push(format!("var({})", zn[r]));
            }
        }
        Cov::Ident => {
            est.push(format!("lns{}_1_1:_cons", l + 1));
            disp.push(format!("var({})", zn.join(" ")));
        }
        Cov::Exch => {
            est.push(format!("lns{}_1_1:_cons", l + 1));
            disp.push(format!("var({})", zn.join(" ")));
            if q > 1 {
                est.push(format!("atr{}_1_1_1:_cons", l + 1));
                disp.push(format!("cov({})", zn.join(" ")));
            }
        }
        Cov::Uns => {
            for r in 0..q {
                est.push(format!("lns{}_1_{}:_cons", l + 1, r + 1));
                disp.push(format!("var({})", zn[r]));
            }
            for r in 0..q {
                for s in r + 1..q {
                    est.push(format!("atr{}_1_{}_{}:_cons", l + 1, r + 1, s + 1));
                    disp.push(format!("cov({},{})", zn[r], zn[s]));
                }
            }
        }
    }
    (est, disp)
}

pub struct Mixed;

struct LevelSpec {
    var: String,
    revars: Vec<String>,
    cov: Cov,
}

impl CommandHandler for Mixed {
    fn name(&self) -> &'static str {
        "mixed"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let raw = cmd.raw.trim();
        let after_verb = raw.split_once(char::is_whitespace).map(|x| x.1).unwrap_or("").trim();
        if !raw.contains("||") {
            if st.e.macros.get("cmd").map(String::as_str) == Some("mixed") && cmd.main.trim().is_empty() {
                est::stash_table_from_e(&st.e, st.level);
                return Ok(st.e.macros.get("_output").cloned().unwrap_or_default());
            }
            return Err(if cmd.main.trim().is_empty() { rc_err(301, "last estimates not found") } else { rc_err(198, "random-effects equation required (||)") });
        }
        let mut parts = after_verb.split("||");
        let fixed = parts.next().unwrap_or("").trim().to_string();
        let mut global_opts: Vec<String> = Vec::new();
        let mut specs = Vec::new();
        for part in parts {
            let (body, opts) = match part.split_once(',') {
                Some((a, b)) => (a.trim(), b.trim()),
                None => (part.trim(), ""),
            };
            let (gv, rest) = body.split_once(':').ok_or_else(|| rc_err(198, "invalid random-effects specification"))?;
            let gv = gv.trim();
            if gv == "_all" {
                return Err(rc_err(198, "_all: crossed random effects not supported"));
            }
            let var = crate::varlist::resolve_name(ds, gv)?;
            let mut revars = if rest.trim().is_empty() { vec![] } else { crate::varlist::expand(ds, rest.trim())? };
            let lopts = ferrostat_parser::parser::parse_line(&format!("x , {opts}")).map_err(|e| rc_err(198, e.message))?;
            let mut cons = true;
            let mut cov: Option<Cov> = None;
            for (n, a) in &lopts.opts {
                let nl = n.to_lowercase();
                if nl.len() >= 5 && "noconstant".starts_with(nl.as_str()) {
                    cons = false;
                } else if nl.len() >= 3 && "covariance".starts_with(nl.as_str()) {
                    cov = Some(Cov::parse(a.as_deref().unwrap_or(""))?);
                } else {
                    global_opts.push(match a {
                        Some(a) => format!("{n}({a})"),
                        None => n.clone(),
                    });
                }
            }
            if cons {
                revars.push("_cons".into());
            }
            if revars.is_empty() {
                return Err(rc_err(198, "no random effects specified"));
            }
            // a lone random intercept is reported with an identity structure
            let cov = cov.unwrap_or(if revars.len() == 1 { Cov::Ident } else { Cov::Ind });
            specs.push(LevelSpec { var, revars, cov });
        }
        let fixed_line = if global_opts.is_empty() {
            format!("mixed {fixed}")
        } else if fixed.contains(',') {
            format!("mixed {fixed} {}", global_opts.join(" "))
        } else {
            format!("mixed {fixed}, {}", global_opts.join(" "))
        };
        let fcmd = ferrostat_parser::parser::parse_line(&fixed_line).map_err(|e| rc_err(198, e.message))?;
        if fcmd.weight.is_some() {
            return Err(rc_err(101, "weights not allowed"));
        }
        let reml = fcmd.has_opt("reml", 4);
        let main = fcmd.main.trim().to_string();
        let (dep, indep) = main.split_once(char::is_whitespace).unwrap_or((main.as_str(), ""));
        let dep = crate::varlist::resolve_name(ds, dep)?;
        let mut extra: Vec<String> = specs.iter().map(|s| s.var.clone()).collect();
        for s in &specs {
            extra.extend(s.revars.iter().filter(|v| *v != "_cons").cloned());
        }
        let d = est::build(ds, st, &fcmd, DesignSpec { depvar: Some(&dep), indep, noconstant: fcmd.has_opt("noconstant", 3), weights: "", extra })?;
        let level = util::level(&fcmd, st)?;
        let n = d.n();
        let act = d.active();
        let p = act.len();
        if n <= p {
            return Err(rc_err(2001, "insufficient observations"));
        }
        let ivars: Vec<String> = specs.iter().map(|s| s.var.clone()).collect();
        let revars: Vec<Vec<String>> = specs.iter().map(|s| s.revars.clone()).collect();
        let stc = structure(ds, &d.rows, &ivars, &revars)?;
        let cols: Vec<&[f64]> = act.iter().map(|&j| d.cols[j].as_slice()).collect();
        let covs: Vec<Cov> = specs.iter().map(|s| s.cov).collect();
        let model = Mx::new(&d.y, cols.clone(), &stc, covs.clone(), reml);
        let nth = model.ntheta();
        // linear model (all random effects zero) for the LR test
        let zero_psis: Vec<Matrix> = (0..specs.len()).map(|l| Matrix::zeros(model.q(l), model.q(l))).collect();
        let ll_c = {
            let acc = model.acc(&zero_psis, 1.0).ok_or_else(|| rc_err(430, "linear model could not be fitted"))?;
            let (_, rss, ld) = Mx::gls(&acc).ok_or_else(|| rc_err(430, "linear model could not be fitted"))?;
            let nf = n as f64;
            let l2p = (2.0 * std::f64::consts::PI).ln();
            if reml {
                let dd = nf - p as f64;
                -0.5 * (dd * (1.0 + l2p + (rss / dd).ln()) + ld)
            } else {
                -0.5 * (nf * (1.0 + l2p + (rss / nf).ln()))
            }
        };
        // optimisation of the relative parameters
        let prof = |th: &[f64]| model.profiled(th);
        let nl = NumLik { f: &prof, k: nth };
        let mut th0 = vec![0.0; nth];
        {
            // a rough start: try a few common relative scales
            let mut best = (f64::NEG_INFINITY, 0.0);
            for s in [-1.5f64, -0.75, 0.0, 0.75] {
                let mut t = vec![0.0; nth];
                let mut o = 0;
                for (l, c) in covs.iter().enumerate() {
                    for (k, isl) in c.is_lnsd(model.q(l)).iter().enumerate() {
                        if *isl {
                            t[o + k] = s;
                        }
                    }
                    o += c.npar(model.q(l));
                }
                let v = prof(&t);
                if v > best.0 {
                    best = (v, s);
                    th0 = t;
                }
            }
        }
        let res = crate::choice_mixed::maximize_tight(&nl, th0, 200)?;
        let th = res.b.clone();
        let s2 = model.sigma2(&th);
        let lnse = 0.5 * s2.ln();
        // absolute parameters
        let mut phi = th.clone();
        {
            let mut o = 0;
            for (l, c) in covs.iter().enumerate() {
                for (k, isl) in c.is_lnsd(model.q(l)).iter().enumerate() {
                    if *isl {
                        phi[o + k] += lnse;
                    }
                }
                o += c.npar(model.q(l));
            }
        }
        phi.push(lnse);
        let m = phi.len();
        let psis = model.psis(&phi[..m - 1], 0.0);
        let acc = model.acc(&psis, s2).ok_or_else(|| rc_err(430, "could not compute the variance at the optimum"))?;
        let (beta, _, _) = Mx::gls(&acc).ok_or_else(|| rc_err(430, "could not compute the variance at the optimum"))?;
        // information matrix: the fixed effects and the covariance
        // parameters jointly (ML), or the covariance parameters alone (REML,
        // where the fixed effects get (X'V^-1 X)^-1)
        // derivatives are taken on the residuals at the optimum (shifting y
        // by X b leaves the GLS problem unchanged and avoids cancellation)
        let r0: Vec<f64> = (0..n).map(|t| d.y[t] - (0..p).map(|j| cols[j][t] * beta[j]).sum::<f64>()).collect();
        let zero_b = vec![0.0; p];
        let rm = Mx::new(&r0, cols.clone(), &stc, covs.clone(), reml);
        let hphi = if reml {
            let f = |ph: &[f64]| rm.full_reml(ph);
            num_grad_hess(&f, &phi, 2e-4).2
        } else {
            let f = |ph: &[f64]| rm.full_ml(&zero_b, ph);
            num_grad_hess(&f, &phi, 2e-4).2
        };
        let mut info = Matrix::zeros(p + m, p + m);
        for a in 0..p {
            for c in 0..p {
                info[(a, c)] = acc.xvx[(a, c)];
            }
        }
        for a in 0..m {
            for c in 0..m {
                info[(p + a, p + c)] = -hphi[(a, c)];
            }
        }
        if !reml {
            // cross derivatives d/dphi of the beta score X'V^-1 (y - X b)
            let gb = |ph: &[f64]| -> Vec<f64> {
                let psis = rm.psis(&ph[..m - 1], 0.0);
                let s2 = (2.0 * ph[m - 1]).exp();
                match rm.acc(&psis, s2) {
                    Some(a) => (0..p).map(|i| a.xvy[i]).collect(),
                    None => vec![f64::NAN; p],
                }
            };
            for j in 0..m {
                let h = 1e-5;
                let mut pp = phi.clone();
                pp[j] += h;
                let gp = gb(&pp);
                pp[j] -= 2.0 * h;
                let gm = gb(&pp);
                for i in 0..p {
                    let d2 = -(gp[i] - gm[i]) / (2.0 * h);
                    info[(i, p + j)] = d2;
                    info[(p + j, i)] = d2;
                }
            }
        }
        let (vfull, _) = info.sym_inverse(1e-14);
        let (vbeta, vphi) = if reml {
            let (vb, _) = acc.xvx.sym_inverse(1e-14);
            let (vp, _) = hphi.scale(-1.0).sym_inverse(1e-14);
            (vb, vp)
        } else {
            let mut vb = Matrix::zeros(p, p);
            let mut vp = Matrix::zeros(m, m);
            for a in 0..p {
                for c in 0..p {
                    vb[(a, c)] = vfull[(a, c)];
                }
            }
            for a in 0..m {
                for c in 0..m {
                    vp[(a, c)] = vfull[(p + a, p + c)];
                }
            }
            (vb, vp)
        };
        let ll = res.ll;
        // ---- posting ----
        let kt = d.k();
        let mut names: Vec<String> = d.names.iter().map(|nm| format!("{dep}:{nm}")).collect();
        let mut disp: Vec<(usize, String)> = Vec::new();
        for (l, s) in specs.iter().enumerate() {
            let (e, dn) = re_names(l, s.cov, &s.revars);
            names.extend(e);
            disp.extend(dn.into_iter().map(|x| (l, x)));
        }
        names.push("lnsig_e:_cons".into());
        let total = kt + m;
        let mut b = vec![0.0; total];
        let mut idx: Vec<usize> = act.clone();
        for (a, &j) in act.iter().enumerate() {
            b[j] = beta[a];
        }
        for r in 0..m {
            b[kt + r] = phi[r];
            idx.push(kt + r);
        }
        let mut vt = Matrix::zeros(p + m, p + m);
        for a in 0..p {
            for c in 0..p {
                vt[(a, c)] = vbeta[(a, c)];
            }
        }
        if !reml {
            vt = vfull.clone();
        }
        for a in 0..m {
            for c in 0..m {
                vt[(p + a, p + c)] = vphi[(a, c)];
            }
        }
        let v = est::expand_v(&vt, &idx, total);
        let vfix = est::expand_v(&vbeta, &act, kt);
        let (chi2, df_m) = est::wald_all(&d, &b[..kt], &vfix);
        let pval = dist::chi2tail(df_m, chi2);
        let chi2_c = (2.0 * (ll - ll_c)).max(0.0);
        let df_c = nth as f64;
        let p_c = if nth == 1 { 0.5 * dist::chi2tail(1.0, chi2_c) } else { dist::chi2tail(df_c, chi2_c) };
        // iteration log
        let ll_label = if reml { "Log restricted-likelihood" } else { "Log likelihood" };
        let mut out = vec![String::new(), "Performing EM optimization ...".to_string(), String::new(), "Performing gradient-based optimization: ".to_string()];
        out.extend(est::iteration_log(&res.iterations, ll_label));
        out.push(String::new());
        out.push("Computing standard errors ...".into());
        let head = out.join("\n");
        let mut omitted = d.omitted.clone();
        omitted.extend(vec![false; m]);
        let mut is_base = d.is_base.clone();
        is_base.extend(vec![false; m]);
        let mut dpost = d.clone();
        dpost.names = names.clone();
        dpost.omitted = omitted;
        dpost.is_base = is_base;
        let ngroups: Vec<Vec<usize>> = stc.sizes.clone();
        let respec: Vec<String> = specs.iter().map(|s| s.revars.join(" ")).collect();
        let covkeys: Vec<&str> = specs.iter().map(|s| s.cov.key()).collect();
        let disp_names: Vec<String> = disp.iter().map(|(l, s)| format!("{}|{}", l, s)).collect();
        est::post(st, "mixed", &dpost, &b, &v, &|e: &mut Results| {
            e.set("ll", ll);
            e.set("ll_c", ll_c);
            e.set("chi2", chi2);
            e.set("df_m", df_m);
            e.set("p", pval);
            e.set("chi2_c", chi2_c);
            e.set("df_c", df_c);
            e.set("p_c", p_c);
            e.set("k", (p + m) as f64);
            e.set("k_f", p as f64);
            e.set("k_r", m as f64);
            e.set("k_rs", nth as f64);
            e.set("k_res", 1.0);
            e.set("rank", (p + m) as f64);
            e.set("sigma_e", s2.sqrt());
            e.set("converged", if res.converged { 1.0 } else { 0.0 });
            e.set("ic", (res.iterations.len() - 1) as f64);
            let nl = ngroups.len();
            let mut ng = Matrix::zeros(1, nl);
            let mut mn = Matrix::zeros(1, nl);
            let mut av = Matrix::zeros(1, nl);
            let mut mx = Matrix::zeros(1, nl);
            for (l, c) in ngroups.iter().enumerate() {
                ng[(0, l)] = c.len() as f64;
                mn[(0, l)] = *c.iter().min().unwrap_or(&0) as f64;
                av[(0, l)] = c.iter().sum::<usize>() as f64 / c.len().max(1) as f64;
                mx[(0, l)] = *c.iter().max().unwrap_or(&0) as f64;
            }
            e.set_mat("N_g", ng);
            e.set_mat("g_min", mn);
            e.set_mat("g_avg", av);
            e.set_mat("g_max", mx);
            for (l, ps) in psis.iter().enumerate() {
                e.set_mat(&format!("_Psi{}", l + 1), ps.clone());
            }
            e.set_str("ivars", ivars.join(" "));
            e.set_str("revars", respec.join(" "));
            e.set_str("_respec", respec.join(";"));
            e.set_str("vartypes", covkeys.join(" "));
            e.set_str("_disp", disp_names.join(";"));
            e.set_str("method", if reml { "REML" } else { "ML" });
            e.set_str("title", "Mixed-effects regression");
            e.set_str("chi2type", "Wald");
            e.set_str("vce", "oim");
            e.set_str("predict", "mixed_p");
            e.set_str("_head", head.clone());
        });
        let text = format!("{}\n{}", head, render(&st.e, level));
        st.e.set_str("_output", text.clone());
        Ok(text)
    }
}

/// Header, fixed-effects table and variance-components table from e().
fn render(e: &Results, level: f64) -> String {
    let get = |s: &str| e.get(s).unwrap_or(SYSMISS);
    let b = e.matrices.get("b").cloned().unwrap_or_else(|| Matrix::zeros(1, 0));
    let v = e.matrices.get("V").cloned().unwrap_or_else(|| Matrix::zeros(0, 0));
    let dep = e.macros.get("depvar").cloned().unwrap_or_default();
    let reml = e.macros.get("method").is_some_and(|m| m == "REML");
    let ivars: Vec<String> = e.macros.get("ivars").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let mat = |s: &str| e.matrices.get(s).map(|m| m.data.clone()).unwrap_or_default();
    let (ng, mn, av, mx) = (mat("N_g"), mat("g_min"), mat("g_avg"), mat("g_max"));
    let row = hrow;
    let mut out = vec![String::new()];
    let title = if reml { "Mixed-effects REML regression" } else { "Mixed-effects ML regression" };
    out.push(row(title, "Number of obs", util::commas(get("N") as usize)));
    if ivars.len() == 1 {
        out.push(row(&format!("Group variable: {}", ivars[0]), "Number of groups", util::commas(ng.first().copied().unwrap_or(0.0) as usize)));
        out.push(format!("{:<48}Obs per group:", ""));
        out.push(format!("{:<48}{:>17} = {:>10}", "", "min", util::commas(mn[0] as usize)));
        out.push(format!("{:<48}{:>17} = {:>10}", "", "avg", format!("{:.1}", av[0])));
        out.push(format!("{:<48}{:>17} = {:>10}", "", "max", util::commas(mx[0] as usize)));
    } else {
        out.push(String::new());
        out.push("-".repeat(61));
        out.push("                |     No. of       Observations per group".into());
        out.push(" Group variable |     groups    Minimum    Average    Maximum".into());
        out.push(format!("{}+{}", "-".repeat(16), "-".repeat(44)));
        for (l, iv) in ivars.iter().enumerate() {
            out.push(format!("{:>15} | {:>10} {:>10} {:>10} {:>10}", crate::summarize::abbrev(iv, 15), util::commas(ng[l] as usize), util::commas(mn[l] as usize), format!("{:.1}", av[l]), util::commas(mx[l] as usize)));
        }
        out.push("-".repeat(61));
        out.push(String::new());
    }
    out.push(row("", &format!("Wald chi2({})", fmt_g(get("df_m"), 9)), f2(get("chi2"))));
    out.push(row(&format!("{} = {}", if reml { "Log restricted-likelihood" } else { "Log likelihood" }, fmt_g(get("ll"), 9)), "Prob > chi2", f4(get("p"))));
    out.push(String::new());
    let names = b.colnames.clone();
    let kf = names.iter().take_while(|n| n.split_once(':').map(|x| x.0) == Some(dep.as_str())).count();
    let fnames: Vec<String> = names[..kf].iter().map(|n| n.split_once(':').map(|x| x.1.to_string()).unwrap_or_else(|| n.clone())).collect();
    let mut fv = Matrix::zeros(kf, kf);
    for i in 0..kf {
        for j in 0..kf {
            fv[(i, j)] = v[(i, j)];
        }
    }
    let is_base: Vec<bool> = fnames.iter().map(|c| c.split('#').all(|p| p.split_once('.').is_some_and(|(l, _)| l.ends_with('b') && l.trim_end_matches('b').parse::<f64>().is_ok()))).collect();
    let omitted: Vec<bool> = fnames.iter().zip(&is_base).map(|(c, ib)| *ib || c.starts_with("o.") || c.contains("o.")).collect();
    out.extend(est::coef_table(&est::TableSpec { depvar: &dep, names: &fnames, b: &b.data[..kf], v: &fv, omitted: &omitted, is_base: &is_base, df: None, level, eform: None, coef_title: "Coefficient" }));
    out.push(String::new());
    out.extend(re_table(e, level));
    let dfc = get("df_c");
    if dfc == 1.0 {
        out.push(format!("LR test vs. linear model: chibar2(01) = {:.2}{:>34}", get("chi2_c"), format!("Prob >= chibar2 = {:.4}", get("p_c"))));
    } else if !is_missing(dfc) {
        out.push(format!("LR test vs. linear model: chi2({}) = {:.2}{:>30}", fmt_g(dfc, 9), get("chi2_c"), format!("Prob > chi2 = {:.4}", get("p_c"))));
        out.push(String::new());
        out.push("Note: LR test is conservative and provided only for reference.".into());
    }
    out.join("\n")
}

/// Rows of the variance-components table: (level or None for the
/// residual, label, estimate, se, lower, upper).
fn re_rows(e: &Results, level: f64) -> Vec<(Option<usize>, String, f64, f64, f64, f64)> {
    let b = e.matrices.get("b").cloned().unwrap_or_else(|| Matrix::zeros(1, 0));
    let v = e.matrices.get("V").cloned().unwrap_or_else(|| Matrix::zeros(0, 0));
    let crit = zcrit(level);
    let disp: Vec<(usize, String)> = e.macros.get("_disp").map(|s| s.split(';').filter(|x| !x.is_empty()).map(|x| {
        let (l, n) = x.split_once('|').unwrap_or(("0", x));
        (l.parse().unwrap_or(0), n.to_string())
    }).collect()).unwrap_or_default();
    let names = &b.colnames;
    let start = names.iter().position(|n| n.starts_with("lns") || n.starts_with("atr")).unwrap_or(names.len());
    let mut rows = Vec::new();
    // parameters of each level in estimation order
    let mut j = start;
    let mut di = 0;
    let lnsd_at = |name: &str| names.iter().position(|n| n == name);
    while j < names.len() && di < disp.len() {
        let nm = &names[j];
        let (l, label) = disp[di].clone();
        if nm.starts_with("lns") {
            let ls = b.data[j];
            let se = v[(j, j)].max(0.0).sqrt();
            let var = (2.0 * ls).exp();
            rows.push((Some(l), label, var, 2.0 * var * se, (2.0 * (ls - crit * se)).exp(), (2.0 * (ls + crit * se)).exp()));
        } else {
            // atr<l>_1_<r>_<s>: covariance from the two sds and the correlation
            let core = nm.trim_end_matches(":_cons");
            let parts: Vec<&str> = core.split('_').collect();
            let lv = parts[0].trim_start_matches("atr");
            let (r, s) = if parts.len() == 4 { (parts[2], parts[3]) } else { ("1", "1") };
            let i1 = lnsd_at(&format!("lns{lv}_1_{r}:_cons")).unwrap_or(j);
            let exch = parts.len() == 4 && r == "1" && s == "1";
            let i2 = if exch { i1 } else { lnsd_at(&format!("lns{lv}_1_{s}:_cons")).unwrap_or(j) };
            let (s1, s2) = (b.data[i1].exp(), b.data[i2].exp());
            let rho = b.data[j].tanh();
            let cov = rho * s1 * s2;
            // gradient in (ln s1, ln s2, atanh rho)
            let gr = [cov, cov, s1 * s2 * (1.0 - rho * rho)];
            let ix = [i1, i2, j];
            let mut var = 0.0;
            for a in 0..3 {
                for c in 0..3 {
                    var += gr[a] * gr[c] * v[(ix[a], ix[c])];
                }
            }
            let se = var.max(0.0).sqrt();
            rows.push((Some(l), label, cov, se, cov - crit * se, cov + crit * se));
        }
        j += 1;
        di += 1;
    }
    if let Some(k) = names.iter().position(|n| n == "lnsig_e:_cons") {
        let ls = b.data[k];
        let se = v[(k, k)].max(0.0).sqrt();
        let var = (2.0 * ls).exp();
        rows.push((None, "var(Residual)".to_string(), var, 2.0 * var * se, (2.0 * (ls - crit * se)).exp(), (2.0 * (ls + crit * se)).exp()));
    }
    rows
}

fn re_table(e: &Results, level: f64) -> Vec<String> {
    let ivars: Vec<String> = e.macros.get("ivars").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let types: Vec<String> = e.macros.get("vartypes").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let bar = "-".repeat(78);
    let mid = format!("{}+{}", "-".repeat(29), "-".repeat(48));
    let mut out = vec![bar.clone(), format!("  Random-effects parameters  |   Estimate   Std. err.     [{}% conf. interval]", fmt_g(level, 4)), mid.clone()];
    let mut last: Option<usize> = None;
    let cell = |x: f64| if is_missing(x) || !x.is_finite() { ".".to_string() } else { fmt_g(x, 8) };
    for (l, label, est_, se, lo, hi) in re_rows(e, level) {
        if l != last {
            if last.is_some() || l.is_none() {
                out.push(mid.clone());
            }
            if let Some(lv) = l {
                let ty = types.get(lv).map(|t| {
                    let mut c = t.chars();
                    c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
                });
                out.push(format!("{:<29}|", format!("{}: {}", ivars.get(lv).cloned().unwrap_or_default(), ty.unwrap_or_default())));
            }
            last = l;
        }
        let label = if label.len() > 28 { crate::summarize::abbrev(&label, 28) } else { label };
        out.push(format!("{:>28} |  {:>9}  {:>9}     {:>9}   {:>9}", label, cell(est_), cell(se), cell(lo), cell(hi)));
    }
    out.push(bar);
    out
}

// ---------------------------------------------------------------------------
// predict / estat
// ---------------------------------------------------------------------------

/// Best linear unbiased predictions of the random effects: per level, per
/// unit, the q predicted effects; plus the structure used.
fn blups(ds: &DataSet, st: &Store, rows: &[usize], resid: &[f64]) -> Result<(Structure, Vec<Vec<Vec<f64>>>), String> {
    let e = &st.e;
    let ivars: Vec<String> = e.macros.get("ivars").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let revars: Vec<Vec<String>> = e.macros.get("_respec").map(|s| s.split(';').map(|x| x.split_whitespace().map(String::from).collect()).collect()).unwrap_or_default();
    let stc = structure(ds, rows, &ivars, &revars)?;
    let psis: Vec<Matrix> = (0..ivars.len()).map(|l| e.matrices.get(&format!("_Psi{}", l + 1)).cloned().unwrap_or_else(|| Matrix::zeros(0, 0))).collect();
    let s2 = e.get("sigma_e").unwrap_or(1.0).powi(2);
    let mut u: Vec<Vec<Vec<f64>>> = stc.sizes.iter().zip(&stc.z).map(|(s, z)| vec![vec![0.0; z.len()]; s.len()]).collect();
    let model = Mx::new(resid, vec![], &stc, vec![], false);
    let lf: Vec<Matrix> = psis.iter().map(psd_factor).collect();
    for (gi, top) in stc.tops.iter().enumerate() {
        let g = &model.pre[gi];
        let nq = g.nq;
        let m = Mx::mfactor(g, &lf, s2).ok_or_else(|| rc_err(430, "could not invert the marginal covariance"))?;
        // Z'V^-1 r = (Z'r - Z'Z L M^-1 L'Z'r / s2) / s2
        let c = Mx::lt_mul(g, &lf, &g.zty, 1);
        let mut d = c.clone();
        fsolve(&m, nq, &mut d);
        bsolve(&m, nq, &mut d);
        let ld = Mx::l_mul(g, &lf, &d);
        let f: Vec<f64> = (0..nq).map(|i| (g.zty[i] - (0..nq).map(|j| g.ztz[i * nq + j] * ld[j]).sum::<f64>() / s2) / s2).collect();
        let ltf = Mx::lt_mul(g, &lf, &f, 1);
        let uhat = Mx::l_mul(g, &lf, &ltf);
        let (offs, _, _) = group_offsets(&stc, top);
        for (&(l, unit), &o) in &offs {
            for r in 0..stc.z[l].len() {
                u[l][unit][r] = uhat[o + r];
            }
        }
    }
    Ok((stc, u))
}

pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let words: Vec<&str> = cmd.main.split_whitespace().collect();
    let names: Vec<String> = match words.first() {
        Some(w) if ferrostat_core::VarType::from_type_str(w).is_some() => words[1..].iter().map(|s| s.to_string()).collect(),
        _ => words.iter().map(|s| s.to_string()).collect(),
    };
    if names.is_empty() {
        return Err(rc_err(100, "new variable name required"));
    }
    let known = ["xb", "stdp", "fitted", "residuals", "reffects", "rstandard"];
    let stat = cmd.opts.iter().map(|(n, _)| n.to_lowercase()).find_map(|n| known.iter().find(|k| k.starts_with(n.as_str()) && n.len() >= 2).map(|k| k.to_string()));
    let msg = if stat.is_none() { "(option xb assumed)".to_string() } else { String::new() };
    let stat = stat.unwrap_or_else(|| "xb".into());
    let b = st.e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let v = st.e.matrices.get("V").cloned().unwrap_or_else(|| Matrix::zeros(0, 0));
    let dep = st.e.macros.get("depvar").cloned().unwrap_or_default();
    let mask = util::touse(ds, st, cmd)?;
    let n = ds.n_obs;
    let mut out_vars: Vec<(String, Vec<f64>)> = Vec::new();
    let xb: Vec<f64> = {
        let env = Env::with_store(ds, st);
        (0..n).map(|i| if mask[i] { crate::postest::xb_row(&env, &b, Some(&dep), i) } else { Ok(SYSMISS) }).collect::<Result<_, _>>()?
    };
    match stat.as_str() {
        "xb" => out_vars.push((names[0].clone(), xb)),
        "stdp" => {
            let env = Env::with_store(ds, st);
            let kf = b.colnames.iter().take_while(|c| c.split_once(':').map(|x| x.0) == Some(dep.as_str())).count();
            let mut vals = vec![SYSMISS; n];
            for i in 0..n {
                if is_missing(xb[i]) {
                    continue;
                }
                let x: Vec<f64> = b.colnames[..kf].iter().map(|c| crate::postest::coef_value(&env, c, i)).collect::<Result<_, _>>()?;
                vals[i] = (0..kf).map(|a| (0..kf).map(|c| x[a] * v[(a, c)] * x[c]).sum::<f64>()).sum::<f64>().max(0.0).sqrt();
            }
            out_vars.push((names[0].clone(), vals));
        }
        _ => {
            // random effects are predicted from the estimation sample
            let sample = st.e.sample.clone().ok_or_else(|| rc_err(301, "estimation sample not found"))?;
            let y = util::col(ds, &dep)?;
            let xs: Vec<f64> = {
                let env = Env::with_store(ds, st);
                (0..n).map(|i| if sample.get(i).copied().unwrap_or(false) { crate::postest::xb_row(&env, &b, Some(&dep), i) } else { Ok(SYSMISS) }).collect::<Result<_, _>>()?
            };
            let rows: Vec<usize> = (0..n).filter(|&i| !is_missing(xs[i]) && !is_missing(y[i])).collect();
            let resid: Vec<f64> = rows.iter().map(|&i| y[i] - xs[i]).collect();
            let (stc, u) = blups(ds, st, &rows, &resid)?;
            match stat.as_str() {
                "reffects" => {
                    let total: usize = stc.z.iter().map(|z| z.len()).sum();
                    let newnames: Vec<String> = if names.len() == 1 && names[0].ends_with('*') {
                        let s = names[0].trim_end_matches('*');
                        (1..=total).map(|i| format!("{s}{i}")).collect()
                    } else {
                        names.clone()
                    };
                    if newnames.len() != total {
                        return Err(rc_err(198, format!("{total} new variable names required")));
                    }
                    let mut k = 0;
                    for (l, z) in stc.z.iter().enumerate() {
                        for r in 0..z.len() {
                            let mut vals = vec![SYSMISS; n];
                            for (t, &i) in rows.iter().enumerate() {
                                if mask[i] {
                                    vals[i] = u[l][stc.ids[l][t]][r];
                                }
                            }
                            out_vars.push((newnames[k].clone(), vals));
                            k += 1;
                        }
                    }
                }
                _ => {
                    let mut vals = vec![SYSMISS; n];
                    let s_e = st.e.get("sigma_e").unwrap_or(1.0);
                    for (t, &i) in rows.iter().enumerate() {
                        if !mask[i] {
                            continue;
                        }
                        let mut zu = 0.0;
                        for (l, z) in stc.z.iter().enumerate() {
                            for r in 0..z.len() {
                                zu += z[r][t] * u[l][stc.ids[l][t]][r];
                            }
                        }
                        let fit = xb[i] + zu;
                        vals[i] = match stat.as_str() {
                            "fitted" => fit,
                            "residuals" => y[i] - fit,
                            _ => (y[i] - fit) / s_e,
                        };
                    }
                    out_vars.push((names[0].clone(), vals));
                }
            }
        }
    }
    let mut lines = Vec::new();
    if !msg.is_empty() {
        lines.push(msg);
    }
    for (nm, vals) in out_vars {
        crate::varlist::check_name(&nm)?;
        if ds.var_index.contains_key(&nm) {
            return Err(rc_err(110, format!("variable {nm} already defined")));
        }
        let nmiss = vals.iter().filter(|v| is_missing(**v)).count();
        ds.add_variable(util::new_double(&nm, vals)).map_err(|e| rc_err(110, e))?;
        if nmiss > 0 {
            lines.push(format!("({} missing value{} generated)", util::commas(nmiss), if nmiss == 1 { "" } else { "s" }));
        }
    }
    Ok(lines.join("\n"))
}

/// `estat icc`, `estat group`, `estat recovariance` after mixed.
pub fn estat(_ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Result<String, String> {
    let level = util::level(cmd, st)?;
    let e = st.e.clone();
    let ivars: Vec<String> = e.macros.get("ivars").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
    match sub {
        "group" => {
            let mat = |s: &str| e.matrices.get(s).map(|m| m.data.clone()).unwrap_or_default();
            let (ng, mn, av, mx) = (mat("N_g"), mat("g_min"), mat("g_avg"), mat("g_max"));
            let mut out = vec!["-".repeat(61), "                |     No. of       Observations per group".into(), " Group variable |     groups    Minimum    Average    Maximum".into(), format!("{}+{}", "-".repeat(16), "-".repeat(44))];
            for (l, iv) in ivars.iter().enumerate() {
                out.push(format!("{:>15} | {:>10} {:>10} {:>10} {:>10}", crate::summarize::abbrev(iv, 15), util::commas(ng[l] as usize), util::commas(mn[l] as usize), format!("{:.1}", av[l]), util::commas(mx[l] as usize)));
            }
            out.push("-".repeat(61));
            Ok(out.join("\n"))
        }
        "recovariance" => {
            let respec: Vec<Vec<String>> = e.macros.get("_respec").map(|s| s.split(';').map(|x| x.split_whitespace().map(String::from).collect()).collect()).unwrap_or_default();
            let mut out = Vec::new();
            for (l, iv) in ivars.iter().enumerate() {
                let ps = e.matrices.get(&format!("_Psi{}", l + 1)).cloned().unwrap_or_else(|| Matrix::zeros(0, 0));
                let zn = &respec[l];
                out.push(String::new());
                out.push(format!("Random-effects covariance matrix for level {iv}"));
                out.push(String::new());
                out.push(format!("             |{}", zn.iter().map(|z| format!(" {:>10}", crate::summarize::abbrev(z, 10))).collect::<String>()));
                out.push(format!("-------------+{}", "-".repeat(11 * zn.len())));
                for r in 0..zn.len() {
                    out.push(format!("{:>12} |{}", crate::summarize::abbrev(&zn[r], 12), (0..=r).map(|c| format!(" {:>10}", fmt_g(ps[(r, c)], 8))).collect::<String>()));
                }
                let mut r = Results::new();
                r.set_mat(&format!("cov{}", l + 1), ps.clone().with_names(zn.clone(), zn.clone()));
                st.r = r;
            }
            Ok(out.join("\n"))
        }
        _ => {
            // icc: random-intercept models only
            let respec: Vec<String> = e.macros.get("_respec").map(|s| s.split(';').map(String::from).collect()).unwrap_or_default();
            if respec.iter().any(|r| r.trim() != "_cons") {
                return Err(rc_err(198, "estat icc is not allowed with random coefficients"));
            }
            let b = e.matrices.get("b").cloned().unwrap_or_else(|| Matrix::zeros(1, 0));
            let v = e.matrices.get("V").cloned().unwrap_or_else(|| Matrix::zeros(0, 0));
            let lidx: Vec<usize> = (0..ivars.len()).filter_map(|l| b.col_index(&format!("lns{}_1_1:_cons", l + 1))).collect();
            let ie = b.col_index("lnsig_e:_cons").ok_or_else(|| rc_err(321, "requested action not valid"))?;
            let var: Vec<f64> = lidx.iter().map(|&j| (2.0 * b.data[j]).exp()).collect();
            let ve = (2.0 * b.data[ie]).exp();
            let tot: f64 = var.iter().sum::<f64>() + ve;
            let mut all = lidx.clone();
            all.push(ie);
            let crit = zcrit(level);
            let mut out = vec![String::new(), "Residual intraclass correlation".to_string(), String::new(), "-".repeat(78), format!("{:>28} |        ICC   Std. err.     [{}% conf. interval]", "Level", fmt_g(level, 4)), format!("{}+{}", "-".repeat(29), "-".repeat(48))];
            let mut r = Results::new();
            let nl = ivars.len();
            let mut icc_v = Vec::new();
            for l in 0..nl {
                // icc at level l: share of variance from levels 0..=l
                let num: f64 = var[..=l].iter().sum();
                let icc = num / tot;
                // gradient wrt the log sds
                let mut g = vec![0.0; all.len()];
                for (k, vk) in var.iter().enumerate() {
                    let dv = 2.0 * vk;
                    g[k] = if k <= l { dv * (tot - num) / (tot * tot) } else { -dv * num / (tot * tot) };
                }
                g[nl] = -2.0 * ve * num / (tot * tot);
                let mut s2 = 0.0;
                for a in 0..all.len() {
                    for c in 0..all.len() {
                        s2 += g[a] * g[c] * v[(all[a], all[c])];
                    }
                }
                let se = s2.max(0.0).sqrt();
                let lg = (icc / (1.0 - icc)).ln();
                let sl = se / (icc * (1.0 - icc));
                let lo = dist::invlogit(lg - crit * sl);
                let hi = dist::invlogit(lg + crit * sl);
                let label = ivars[..=l].iter().rev().cloned().collect::<Vec<_>>().join("|");
                out.push(format!("{:>28} |  {:>9}  {:>9}     {:>9}   {:>9}", crate::summarize::abbrev(&label, 28), fmt_g(icc, 8), fmt_g(se, 8), fmt_g(lo, 8), fmt_g(hi, 8)));
                r.set(&format!("icc{}", l + 2), icc);
                r.set(&format!("se{}", l + 2), se);
                icc_v.push(icc);
            }
            out.push("-".repeat(78));
            r.set("level", level);
            st.r = r;
            Ok(out.join("\n"))
        }
    }
}

#[allow(dead_code)]
fn unused(_: &dyn Likelihood) {}
