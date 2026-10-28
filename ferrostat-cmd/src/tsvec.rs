//! Vector error-correction models by Johansen's maximum likelihood method:
//! `vec`, `vecrank` (trace and maximum-eigenvalue statistics with the
//! Osterwald-Lenum critical values), `veclmar`, `vecnorm` and `vecstable`.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::{fmt_e, fmt_g};
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::tsla::{chol, inv, kron, mul, sym_eig, sym_ginv, tr};
use crate::tsvar::{self, canon, cross, eqname, lagname, ols_system, series, ts, CoefRow, RowKind, Ts};
use crate::util;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Trend {
    None,
    RConstant,
    Constant,
    RTrend,
    Trend,
}

impl Trend {
    pub fn parse(s: Option<&str>) -> Result<Trend, String> {
        Ok(match s.map(|x| x.trim().to_lowercase()).as_deref() {
            None | Some("constant") | Some("con") | Some("cons") => Trend::Constant,
            Some("rconstant") | Some("rcon") => Trend::RConstant,
            Some("trend") => Trend::Trend,
            Some("rtrend") => Trend::RTrend,
            Some("none") => Trend::None,
            Some(o) => return Err(rc_err(198, format!("trend({o}) invalid"))),
        })
    }
    pub fn name(&self) -> &'static str {
        match self {
            Trend::None => "none",
            Trend::RConstant => "rconstant",
            Trend::Constant => "constant",
            Trend::RTrend => "rtrend",
            Trend::Trend => "trend",
        }
    }
    fn title(&self) -> &'static str {
        match self {
            Trend::None => "None",
            Trend::RConstant => "Rconstant",
            Trend::Constant => "Constant",
            Trend::RTrend => "Rtrend",
            Trend::Trend => "Trend",
        }
    }
    /// Extra restricted rows of β̃ (constant or trend).
    fn m1_extra(&self) -> usize {
        matches!(self, Trend::RConstant | Trend::RTrend) as usize
    }
    /// Unrestricted deterministic terms in Z2: (constant, trend).
    fn z2_det(&self) -> (bool, bool) {
        match self {
            Trend::None | Trend::RConstant => (false, false),
            Trend::Constant | Trend::RTrend => (true, false),
            Trend::Trend => (true, true),
        }
    }
}

/// Sample of a VEC model with `p` lags in the levels VAR.
pub struct VData {
    pub rows: Vec<usize>,
    pub times: Vec<f64>,
    /// levels y_{t-1}
    pub ylag: Vec<Vec<f64>>,
    /// Δy_t
    pub dy: Vec<Vec<f64>>,
    /// Δy_{t-1}, ..., Δy_{t-p+1}, variable-major (x1 lags, x2 lags, ...)
    pub dlags: Vec<Vec<f64>>,
    /// trend index t = time - tmin + 1
    pub trend: Vec<f64>,
}

pub fn build(ds: &DataSet, st: &Store, tsd: &Ts, endog: &[String], p: usize, mask: &[bool]) -> Result<VData, String> {
    let cols: Vec<Vec<f64>> = endog.iter().map(|e| series(ds, st, e)).collect::<Result<_, _>>()?;
    let mut rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask.get(i).copied().unwrap_or(false) && !is_missing(tsd.time[i])).collect();
    rows.sort_by(|&a, &b| tsd.time[a].total_cmp(&tsd.time[b]));
    let mut d = VData { rows: Vec::new(), times: Vec::new(), ylag: Vec::new(), dy: Vec::new(), dlags: Vec::new(), trend: Vec::new() };
    'row: for &r in &rows {
        // levels y_t, ..., y_{t-p}
        let mut lv: Vec<Vec<f64>> = Vec::new();
        for c in &cols {
            let v: Vec<f64> = (0..=p as i64).map(|l| tsd.lag(c, r, l)).collect();
            if v.iter().any(|x| is_missing(*x)) {
                continue 'row;
            }
            lv.push(v);
        }
        d.rows.push(r);
        d.times.push(tsd.time[r]);
        d.ylag.push(lv.iter().map(|v| v[1]).collect());
        d.dy.push(lv.iter().map(|v| v[0] - v[1]).collect());
        d.dlags.push(lv.iter().flat_map(|v| (1..p).map(move |l| v[l] - v[l + 1])).collect());
    }
    if d.rows.is_empty() {
        return Err(rc_err(2000, "no observations"));
    }
    if d.times.windows(2).any(|w| (w[1] - w[0] - 1.0).abs() > 1e-9) {
        return Err(rc_err(498, "gaps not allowed"));
    }
    let t0 = d.times[0];
    d.trend = d.times.iter().map(|t| t - t0 + 1.0).collect();
    Ok(d)
}

fn hcat(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter().zip(b).map(|(x, y)| x.iter().chain(y.iter()).copied().collect()).collect()
}

/// Z1 and Z2 of the concentrated likelihood.
fn zmats(d: &VData, trend: Trend) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let t = d.rows.len();
    let z1: Vec<Vec<f64>> = (0..t)
        .map(|i| {
            let mut r = d.ylag[i].clone();
            match trend {
                Trend::RConstant => r.push(1.0),
                Trend::RTrend => r.push(d.trend[i]),
                _ => {}
            }
            r
        })
        .collect();
    let (c, tt) = trend.z2_det();
    let z2: Vec<Vec<f64>> = (0..t)
        .map(|i| {
            let mut r = d.dlags[i].clone();
            if tt {
                r.push(d.trend[i]);
            }
            if c {
                r.push(1.0);
            }
            r
        })
        .collect();
    (z1, z2)
}

/// Result of the reduced-rank regression.
pub struct Johansen {
    pub lambda: Vec<f64>,
    /// eigenvectors (m1 × m1), normalised v' S11 v = 1
    pub v: Matrix,
    pub s00: Matrix,
    pub s01: Matrix,
    pub s11: Matrix,
    pub t: usize,
}

fn resid_on(z: &[Vec<f64>], x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if x.first().map_or(true, |r| r.is_empty()) {
        return z.to_vec();
    }
    ols_system(z, x).2
}

pub fn johansen(d: &VData, trend: Trend) -> Result<Johansen, String> {
    let (z1, z2) = zmats(d, trend);
    let t = d.rows.len();
    let r0 = resid_on(&d.dy, &z2);
    let r1 = resid_on(&z1, &z2);
    let tf = t as f64;
    let s00 = cross(&r0, tf);
    let s11 = cross(&r1, tf);
    let k0 = r0[0].len();
    let k1 = r1[0].len();
    let mut s01 = Matrix::zeros(k0, k1);
    for i in 0..t {
        for a in 0..k0 {
            for b in 0..k1 {
                s01[(a, b)] += r0[i][a] * r1[i][b] / tf;
            }
        }
    }
    let l = chol(&s11);
    let li = inv(&l).ok_or_else(|| rc_err(506, "matrix not positive definite"))?;
    let s00i = inv(&s00).ok_or_else(|| rc_err(506, "matrix not positive definite"))?;
    let m = mul(&mul(&mul(&mul(&li, &tr(&s01)), &s00i), &s01), &tr(&li));
    let (lam, u) = sym_eig(&m);
    let v = mul(&tr(&li), &u);
    Ok(Johansen { lambda: lam.iter().map(|x| x.clamp(0.0, 1.0 - 1e-15)).collect(), v, s00, s01, s11, t })
}

impl Johansen {
    pub fn ll(&self, r: usize) -> f64 {
        let k = self.s00.rows as f64;
        let t = self.t as f64;
        let s: f64 = self.lambda.iter().take(r).map(|l| (1.0 - l).ln()).sum();
        -t / 2.0 * (k * ((2.0 * std::f64::consts::PI).ln() + 1.0) + self.s00.det().ln() + s)
    }
}

/// Number of freely estimated parameters for rank r.
fn nparms(k: usize, p: usize, trend: Trend, r: usize) -> usize {
    let (c, tt) = trend.z2_det();
    let m2 = k * (p - 1) + c as usize + tt as usize;
    let m1 = k + trend.m1_extra();
    k * m2 + k * r + (m1 - r) * r
}

// ---------------------------------------------------------------------------
// vec
// ---------------------------------------------------------------------------

pub struct VecFit {
    pub endog: Vec<String>,
    pub p: usize,
    pub r: usize,
    pub trend: Trend,
    pub data: VData,
    pub joh: Johansen,
    /// normalised β̃ (m1 × r), α (K × r), Ω (K × K)
    pub beta: Matrix,
    pub alpha: Matrix,
    pub omega: Matrix,
    /// μ and ρ (length r; missing if absent), whether directly estimated
    pub mu: Vec<f64>,
    pub rho: Vec<f64>,
    /// short-run regression
    pub regnames: Vec<String>,
    pub x: Vec<Vec<f64>>,
    pub coef: Vec<Vec<f64>>,
    pub xtxi: Matrix,
    pub resid: Vec<Vec<f64>>,
    pub d: usize,
    pub vbeta: Matrix,
}

pub fn fit(d: VData, endog: &[String], p: usize, r: usize, trend: Trend) -> Result<VecFit, String> {
    let k = endog.len();
    if r < 1 || r >= k {
        return Err(rc_err(198, format!("rank() must be between 1 and {}", k - 1)));
    }
    let joh = johansen(&d, trend)?;
    let m1 = k + trend.m1_extra();
    let t = d.rows.len();
    let tf = t as f64;
    // eigenvectors of the r largest roots, Johansen normalisation
    let mut bt = Matrix::zeros(m1, r);
    for i in 0..m1 {
        for j in 0..r {
            bt[(i, j)] = joh.v[(i, j)];
        }
    }
    let mut top = Matrix::zeros(r, r);
    for i in 0..r {
        for j in 0..r {
            top[(i, j)] = bt[(i, j)];
        }
    }
    let beta = mul(&bt, &inv(&top).ok_or_else(|| rc_err(498, "normalization failed"))?);
    let btsb = mul(&mul(&tr(&beta), &joh.s11), &beta);
    let sigma_b = inv(&btsb).unwrap_or_else(|| Matrix::identity(r));
    let alpha = mul(&mul(&joh.s01, &beta), &sigma_b);
    let omega = crate::tsla::sub(&joh.s00, &mul(&mul(&alpha, &tr(&beta)), &tr(&joh.s01)));
    // predicted non-demeaned cointegrating equations β_y' y_{t-1}
    let ce_raw: Vec<Vec<f64>> = (0..t).map(|i| (0..r).map(|j| (0..k).map(|a| beta[(a, j)] * d.ylag[i][a]).sum()).collect()).collect();
    let mut mu = vec![SYSMISS; r];
    let mut rho = vec![SYSMISS; r];
    match trend {
        Trend::RConstant => mu = (0..r).map(|j| beta[(k, j)]).collect(),
        Trend::RTrend => rho = (0..r).map(|j| beta[(k, j)]).collect(),
        _ => {}
    }
    if matches!(trend, Trend::Constant | Trend::RTrend | Trend::Trend) {
        // auxiliary VAR (11) for the unrestricted deterministic terms
        let ce_aux: Vec<Vec<f64>> = (0..t)
            .map(|i| (0..r).map(|j| ce_raw[i][j] + if trend == Trend::RTrend { rho[j] * d.trend[i] } else { 0.0 }).collect())
            .collect();
        let mut x = hcat(&ce_aux, &d.dlags);
        for (i, row) in x.iter_mut().enumerate() {
            row.push(1.0);
            if trend == Trend::Trend {
                row.push(d.trend[i]);
            }
        }
        let (c, _, _) = ols_system(&d.dy, &x);
        let nreg = x[0].len();
        let ci = if trend == Trend::Trend { nreg - 2 } else { nreg - 1 };
        let ata = mul(&tr(&alpha), &alpha);
        let atai = inv(&ata).unwrap_or_else(|| Matrix::identity(r));
        let proj = |v: &[f64]| -> Vec<f64> { (0..r).map(|j| (0..r).map(|l| atai[(j, l)] * (0..k).map(|a| alpha[(a, l)] * v[a]).sum::<f64>()).sum()).collect() };
        let vhat: Vec<f64> = (0..k).map(|a| c[a][ci]).collect();
        mu = proj(&vhat);
        if trend == Trend::Trend {
            let dhat: Vec<f64> = (0..k).map(|a| c[a][nreg - 1]).collect();
            rho = proj(&dhat);
        }
    }
    // demeaned cointegrating equations and the short-run regression (14)
    let ce: Vec<Vec<f64>> = (0..t)
        .map(|i| (0..r).map(|j| ce_raw[i][j] + if is_missing(mu[j]) { 0.0 } else { mu[j] } + if is_missing(rho[j]) { 0.0 } else { rho[j] * d.trend[i] }).collect())
        .collect();
    let mut regnames: Vec<String> = (1..=r).map(|j| format!("L._ce{j}")).collect();
    for e in endog {
        for l in 1..p {
            regnames.push(lagname(&format!("D.{e}").replace("D.D.", "D2."), l));
        }
    }
    let (c, tt) = match trend {
        Trend::None | Trend::RConstant => (false, false),
        Trend::Constant | Trend::RTrend => (true, false),
        Trend::Trend => (true, true),
    };
    let mut x = hcat(&ce, &d.dlags);
    for (i, row) in x.iter_mut().enumerate() {
        if tt {
            row.push(d.trend[i]);
        }
        if c {
            row.push(1.0);
        }
    }
    if tt {
        regnames.push("_trend".into());
    }
    if c {
        regnames.push("_cons".into());
    }
    let (coef, xtxi, resid) = ols_system(&d.dy, &x);
    let dd = nparms(k, p, trend, r) / k;
    // VCE of β̃ (Johansen 1995): (1/(T-d)) (I⊗H){(α'Ω⁻¹α)⊗(H'S11H)}⁻¹(I⊗H)'
    let m1r = m1 - r;
    let mut h = Matrix::zeros(m1, m1r);
    for i in 0..m1r {
        h[(r + i, i)] = 1.0;
    }
    let oi = inv(&omega).unwrap_or_else(|| Matrix::identity(k));
    let aoa = mul(&mul(&tr(&alpha), &oi), &alpha);
    let hsh = mul(&mul(&tr(&h), &joh.s11), &h);
    let mid = inv(&kron(&aoa, &hsh)).unwrap_or_else(|| Matrix::identity(r * m1r));
    let ih = kron(&Matrix::identity(r), &h);
    let vbeta = mul(&mul(&ih, &mid), &tr(&ih)).scale(1.0 / (tf - dd as f64));
    Ok(VecFit { endog: endog.to_vec(), p, r, trend, data: d, joh, beta, alpha, omega, mu, rho, regnames, x, coef, xtxi, resid, d: dd, vbeta })
}

impl VecFit {
    pub fn k(&self) -> usize {
        self.endog.len()
    }
    pub fn t(&self) -> usize {
        self.data.rows.len()
    }
    pub fn m(&self) -> usize {
        self.regnames.len()
    }
    pub fn ll(&self) -> f64 {
        self.joh.ll(self.r)
    }
    /// Ω with the small-sample divisor (T - d).
    pub fn omega_adj(&self) -> Matrix {
        cross(&self.resid, (self.t() - self.d) as f64)
    }
    pub fn vce(&self) -> Matrix {
        kron(&self.omega_adj(), &self.xtxi)
    }
    pub fn b(&self) -> Vec<f64> {
        self.coef.iter().flatten().copied().collect()
    }
    /// Γ_1..Γ_{p-1} (K × K each).
    pub fn gammas(&self) -> Vec<Matrix> {
        let k = self.k();
        let r = self.r;
        let pm = self.p - 1;
        (0..pm)
            .map(|l| {
                let mut g = Matrix::zeros(k, k);
                for i in 0..k {
                    for j in 0..k {
                        g[(i, j)] = self.coef[i][r + j * pm + l];
                    }
                }
                g
            })
            .collect()
    }
    /// Π = α β_y'.
    pub fn pi(&self) -> Matrix {
        let k = self.k();
        let mut by = Matrix::zeros(k, self.r);
        for i in 0..k {
            for j in 0..self.r {
                by[(i, j)] = self.beta[(i, j)];
            }
        }
        let al = self.short_alpha();
        mul(&al, &tr(&by))
    }
    /// α from the short-run regression.
    pub fn short_alpha(&self) -> Matrix {
        let mut a = Matrix::zeros(self.k(), self.r);
        for i in 0..self.k() {
            for j in 0..self.r {
                a[(i, j)] = self.coef[i][j];
            }
        }
        a
    }
    /// Deterministic terms of the levels VAR: y_t = Σ A_i y_{t-i} + c0 + c1 t.
    pub fn det_terms(&self) -> (Vec<f64>, Vec<f64>) {
        let k = self.k();
        let al = self.short_alpha();
        let m = self.m();
        let mut c0 = vec![0.0; k];
        let mut c1 = vec![0.0; k];
        for i in 0..k {
            for j in 0..self.r {
                if !is_missing(self.mu[j]) {
                    c0[i] += al[(i, j)] * self.mu[j];
                }
                if !is_missing(self.rho[j]) {
                    c1[i] += al[(i, j)] * self.rho[j];
                }
            }
            for (n, name) in self.regnames.iter().enumerate() {
                if name == "_cons" {
                    c0[i] += self.coef[i][n];
                }
                if name == "_trend" {
                    c1[i] += self.coef[i][n];
                }
            }
            let _ = m;
        }
        (c0, c1)
    }
}

/// Levels VAR lag matrices from Π and Γ_i.
pub fn levels_from(pi: &Matrix, gammas: &[Matrix]) -> Vec<Matrix> {
    let k = pi.rows;
    let p = gammas.len() + 1;
    let id = Matrix::identity(k);
    let mut a = Vec::new();
    for i in 1..=p {
        let m = if i == 1 {
            let mut m = crate::tsla::add(pi, &id);
            if p > 1 {
                m = crate::tsla::add(&m, &gammas[0]);
            }
            m
        } else if i < p {
            crate::tsla::sub(&gammas[i - 1], &gammas[i - 2])
        } else {
            gammas[p - 2].scale(-1.0)
        };
        a.push(m);
    }
    a
}

fn mat_from_rows(rows: &[Vec<f64>]) -> Matrix {
    Matrix::from_rows(rows)
}

/// Stored VEC results needed by irf, fcast and vecstable.
pub struct VecRep {
    pub endog: Vec<String>,
    pub a: Vec<Matrix>,
    pub omega: Matrix,
    pub c0: Vec<f64>,
    pub c1: Vec<f64>,
    pub t: f64,
    pub d: f64,
    pub tmin: f64,
    pub rank: usize,
}

pub fn rep_from_e(e: &Results) -> Result<VecRep, String> {
    if e.macros.get("cmd").map(String::as_str) != Some("vec") {
        return Err(rc_err(301, "vec results not found"));
    }
    let endog: Vec<String> = e.macros.get("endog").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let k = endog.len();
    let pi = e.matrices.get("pi").cloned().ok_or_else(|| rc_err(301, "e(pi) not found"))?;
    let g = e.matrices.get("Gamma").cloned().unwrap_or_else(|| Matrix::zeros(k, 0));
    let p = e.get("n_lags").unwrap_or(1.0) as usize;
    let gammas: Vec<Matrix> = (0..p - 1)
        .map(|l| {
            let mut m = Matrix::zeros(k, k);
            for i in 0..k {
                for j in 0..k {
                    m[(i, j)] = g[(i, l * k + j)];
                }
            }
            m
        })
        .collect();
    let det = e.matrices.get("det").cloned().unwrap_or_else(|| Matrix::zeros(k, 2));
    Ok(VecRep {
        a: levels_from(&pi, &gammas),
        omega: e.matrices.get("omega").cloned().ok_or_else(|| rc_err(301, "e(omega) not found"))?,
        c0: (0..k).map(|i| det[(i, 0)]).collect(),
        c1: (0..k).map(|i| det[(i, 1)]).collect(),
        t: e.get("N").unwrap_or(0.0),
        d: e.get("df_dfk").unwrap_or(0.0),
        tmin: e.get("tmin").unwrap_or(0.0),
        rank: e.get("k_ce").unwrap_or(1.0) as usize,
        endog,
    })
}

fn eq_table(f: &VecFit) -> (Vec<String>, Vec<(f64, f64, f64, f64, f64)>) {
    let k = f.k();
    let m = f.m();
    let t = f.t() as f64;
    let om = f.omega_adj();
    let mut stats = Vec::new();
    for i in 0..k {
        let ssr: f64 = f.resid.iter().map(|r| r[i] * r[i]).sum();
        let y: Vec<f64> = f.data.dy.iter().map(|r| r[i]).collect();
        // uncentred: the constant is part of the error-correction structure
        let sst: f64 = y.iter().map(|v| v * v).sum();
        let mut vs = Matrix::zeros(m, m);
        for a in 0..m {
            for b in 0..m {
                vs[(a, b)] = om[(i, i)] * f.xtxi[(a, b)];
            }
        }
        let vi = sym_ginv(&vs);
        let bs = &f.coef[i];
        let chi2: f64 = (0..m).map(|a| (0..m).map(|b| bs[a] * vi[(a, b)] * bs[b]).sum::<f64>()).sum();
        let _ = t;
        stats.push((m as f64, om[(i, i)].sqrt(), 1.0 - ssr / sst, chi2, dist::chi2tail(m as f64, chi2)));
    }
    (f.endog.iter().map(|e| format!("D_{}", eqname(e))).collect(), stats)
}

fn ic(f: &VecFit) -> (f64, f64, f64) {
    let t = f.t() as f64;
    let n = nparms(f.k(), f.p, f.trend, f.r) as f64;
    let ll = f.ll();
    (-2.0 * ll / t + 2.0 * n / t, -2.0 * ll / t + 2.0 * t.ln().ln() / t * n, -2.0 * ll / t + t.ln() / t * n)
}

fn display(f: &VecFit, tsd: &Ts, level: f64, cmd: &Command) -> String {
    let k = f.k();
    let r = f.r;
    let (aic, hqic, sbic) = ic(f);
    let mut out = vec![String::new(), "Vector error-correction model".to_string(), String::new()];
    let line = |l: String, kk: &str, v: String| format!("{l:<48}{kk:<17} = {v:>10}");
    out.push(line(format!("Sample: {} thru {}", tsd.fmt(f.data.times[0]), tsd.fmt(*f.data.times.last().unwrap())), "Number of obs", f.t().to_string()));
    out.push(line(String::new(), "AIC", fmt_g(aic, 8)));
    out.push(line(format!("Log likelihood = {:>9}", fmt_g(f.ll(), 8)), "HQIC", fmt_g(hqic, 8)));
    out.push(line(format!("Det(Sigma_ml)  = {:>9}", fmt_e(f.omega.det(), 2)), "SBIC", fmt_g(sbic, 8)));
    let names_eq: Vec<String> = f.endog.iter().map(|e| format!("D_{}", eqname(e))).collect();
    if !cmd.has_opt("noetable", 4) {
        out.push(String::new());
        out.push(format!("{:<15}{:>9}{:>10}{:>9}{:>10}{:>11}", "Equation", "Parms", "RMSE", "R-sq", "chi2", "P>chi2"));
        out.push("-".repeat(64));
        let (_, stats) = eq_table(f);
        for (i, s) in stats.iter().enumerate() {
            out.push(format!("{:<15}{:>8}{:>12}{:>9.4}{:>11}{:>9.4}", crate::summarize::abbrev(&names_eq[i], 15), s.0, fmt_g(s.1, 7), s.2, fmt_g(s.3, 8), s.4));
        }
        out.push("-".repeat(64));
        out.push(String::new());
        let names: Vec<String> = names_eq.iter().flat_map(|q| f.regnames.iter().map(move |n| format!("{q}:{n}"))).collect();
        let b = f.b();
        let v = f.vce();
        let no = vec![false; names.len()];
        let spec = crate::est::TableSpec { depvar: "", names: &names, b: &b, v: &v, omitted: &no, is_base: &no, df: None, level, eform: None, coef_title: "Coefficient" };
        out.extend(crate::est::coef_table(&spec));
    }
    if cmd.has_opt("alpha", 5) {
        out.push(String::new());
        out.push("Adjustment parameters".into());
        out.push(format!("{:<15}{:>9}{:>12}{:>11}", "Equation", "Parms", "chi2", "P>chi2"));
        out.push("-".repeat(47));
        let v = f.vce();
        let m = f.m();
        let mut rows = Vec::new();
        for i in 0..k {
            let idx: Vec<usize> = (0..r).map(|j| i * m + j).collect();
            let mut vs = Matrix::zeros(r, r);
            for (a, &ia) in idx.iter().enumerate() {
                for (b, &ib) in idx.iter().enumerate() {
                    vs[(a, b)] = v[(ia, ib)];
                }
            }
            let vi = sym_ginv(&vs);
            let bs: Vec<f64> = (0..r).map(|j| f.coef[i][j]).collect();
            let chi2: f64 = (0..r).map(|a| (0..r).map(|b| bs[a] * vi[(a, b)] * bs[b]).sum::<f64>()).sum();
            out.push(format!("{:<15}{:>8}{:>13}{:>11.4}", crate::summarize::abbrev(&names_eq[i], 15), r, fmt_g(chi2, 8), dist::chi2tail(r as f64, chi2)));
            for j in 0..r {
                rows.push(CoefRow { name: format!("{}:L._ce{}", names_eq[i], j + 1), b: f.coef[i][j], se: v[(i * m + j, i * m + j)].sqrt(), kind: RowKind::Normal });
            }
        }
        out.push("-".repeat(47));
        out.push(String::new());
        out.extend(tsvar::coef_rows_table(&rows, None, level, "alpha"));
    }
    if !cmd.has_opt("nobtable", 4) {
        out.push(String::new());
        out.push("Cointegrating equations".into());
        out.push(format!("{:<15}{:>9}{:>12}{:>11}", "Equation", "Parms", "chi2", "P>chi2"));
        out.push("-".repeat(47));
        let m1 = k + f.trend.m1_extra();
        let m1r = m1 - r;
        for j in 0..r {
            // free y-coefficients of β_j: rows r..k
            let idx: Vec<usize> = (r..k).collect();
            let q = idx.len();
            let mut vs = Matrix::zeros(q, q);
            for (a, &ia) in idx.iter().enumerate() {
                for (b, &ib) in idx.iter().enumerate() {
                    vs[(a, b)] = f.vbeta[(j * m1 + ia, j * m1 + ib)];
                }
            }
            let vi = sym_ginv(&vs);
            let bs: Vec<f64> = idx.iter().map(|&i| f.beta[(i, j)]).collect();
            let chi2: f64 = (0..q).map(|a| (0..q).map(|b| bs[a] * vi[(a, b)] * bs[b]).sum::<f64>()).sum();
            out.push(format!("{:<15}{:>8}{:>13}{:>11.4}", format!("_ce{}", j + 1), q, fmt_g(chi2, 8), dist::chi2tail(q as f64, chi2)));
        }
        let _ = m1r;
        out.push("-".repeat(47));
        out.push(String::new());
        out.push("Identification:  beta is exactly identified".into());
        out.push(String::new());
        out.push(format!("                 Johansen normalization restriction{} imposed", if r > 1 { "s" } else { "" }));
        let mut rows = Vec::new();
        for j in 0..r {
            for i in 0..k {
                let val = f.beta[(i, j)];
                let kind = if i < r {
                    if i == j { RowKind::NoSe } else { RowKind::Omitted }
                } else {
                    RowKind::Normal
                };
                rows.push(CoefRow { name: format!("_ce{}:{}", j + 1, f.endog[i]), b: val, se: f.vbeta[(j * m1 + i, j * m1 + i)].max(0.0).sqrt(), kind });
            }
            if f.trend == Trend::RTrend {
                rows.push(CoefRow { name: format!("_ce{}:_trend", j + 1), b: f.rho[j], se: f.vbeta[(j * m1 + k, j * m1 + k)].max(0.0).sqrt(), kind: RowKind::Normal });
            } else if f.trend == Trend::Trend {
                rows.push(CoefRow { name: format!("_ce{}:_trend", j + 1), b: f.rho[j], se: SYSMISS, kind: RowKind::NoSe });
            }
            match f.trend {
                Trend::RConstant => rows.push(CoefRow { name: format!("_ce{}:_cons", j + 1), b: f.mu[j], se: f.vbeta[(j * m1 + k, j * m1 + k)].max(0.0).sqrt(), kind: RowKind::Normal }),
                Trend::None => {}
                _ => rows.push(CoefRow { name: format!("_ce{}:_cons", j + 1), b: f.mu[j], se: SYSMISS, kind: RowKind::NoSe }),
            }
        }
        out.extend(tsvar::coef_rows_table(&rows, None, level, "beta"));
    }
    out.join("\n")
}

fn post(st: &mut Store, f: &VecFit, tsd: &Ts, raw: &str, n: usize) {
    let k = f.k();
    let r = f.r;
    let names_eq: Vec<String> = f.endog.iter().map(|e| format!("D_{}", eqname(e))).collect();
    let names: Vec<String> = names_eq.iter().flat_map(|q| f.regnames.iter().map(move |nm| format!("{q}:{nm}"))).collect();
    let (aic, hqic, sbic) = ic(f);
    let (_, stats) = eq_table(f);
    let m1 = k + f.trend.m1_extra();
    let e = st.begin_e();
    e.set_str("cmd", "vec");
    e.set_str("cmdline", raw.trim());
    e.set_str("title", "Vector error-correction model");
    e.set_str("trend", f.trend.name());
    e.set_str("tsfmt", tsd.tfmt.clone());
    e.set_str("tvar", tsd.tvar.clone());
    e.set_str("endog", f.endog.join(" "));
    e.set_str("eqnames", names_eq.join(" "));
    e.set_str("cenames", (1..=r).map(|j| format!("_ce{j}")).collect::<Vec<_>>().join(" "));
    e.set_str("properties", "b V");
    e.set_str("predict", "vec_p");
    e.set("N", f.t() as f64);
    e.set("k_eq", k as f64);
    e.set("k_dv", k as f64);
    e.set("k_ce", r as f64);
    e.set("n_lags", f.p as f64);
    e.set("df_m", (nparms(k, f.p, f.trend, r)) as f64);
    e.set("k_rank", (nparms(k, f.p, f.trend, r)) as f64);
    e.set("df_dfk", f.d as f64);
    e.set("ll", f.ll());
    e.set("aic", aic);
    e.set("hqic", hqic);
    e.set("sbic", sbic);
    e.set("tmin", f.data.times[0]);
    e.set("tmax", *f.data.times.last().unwrap());
    e.set("detsig_ml", f.omega.det());
    e.set("rank", (k * f.m()) as f64);
    e.set("converge", 1.0);
    e.set("beta_iden", 1.0);
    e.set("beta_icnt", (r * r) as f64);
    for (i, s) in stats.iter().enumerate() {
        e.set(&format!("k_{}", i + 1), s.0);
        e.set(&format!("df_m{}", i + 1), s.0);
        e.set(&format!("rmse_{}", i + 1), s.1);
        e.set(&format!("r2_{}", i + 1), s.2);
        e.set(&format!("chi2_{}", i + 1), s.3);
    }
    let b = f.b();
    e.set_mat("b", Matrix::row_vector(&b).with_names(vec!["y1".into()], names.clone()));
    e.set_mat("V", f.vce().with_names(names.clone(), names));
    // β (row vector over ce:var) and its VCE
    let mut bn = Vec::new();
    let mut bv = Vec::new();
    for j in 0..r {
        for i in 0..k {
            bn.push(format!("_ce{}:{}", j + 1, f.endog[i]));
            bv.push(f.beta[(i, j)]);
        }
        if matches!(f.trend, Trend::RTrend | Trend::Trend) {
            bn.push(format!("_ce{}:_trend", j + 1));
            bv.push(f.rho[j]);
        }
        if f.trend != Trend::None {
            bn.push(format!("_ce{}:_cons", j + 1));
            bv.push(f.mu[j]);
        }
    }
    e.set_mat("beta", Matrix::row_vector(&bv).with_names(vec!["y1".into()], bn.clone()));
    let mut vb = Matrix::zeros(bn.len(), bn.len());
    let per = bn.len() / r;
    for j in 0..r {
        for j2 in 0..r {
            for a in 0..per {
                for c in 0..per {
                    let ia = if a < k { a } else { k };
                    let ic = if c < k { c } else { k };
                    let direct = f.trend.m1_extra() == 1;
                    let ok = (a < k || direct && a == per - 1 || f.trend == Trend::RTrend && a == k) && (c < k || direct && c == per - 1 || f.trend == Trend::RTrend && c == k);
                    vb[(j * per + a, j2 * per + c)] = if ok && ia < m1 && ic < m1 { f.vbeta[(j * m1 + ia, j2 * m1 + ic)] } else { SYSMISS };
                }
            }
        }
    }
    e.set_mat("V_beta", vb.with_names(bn.clone(), bn));
    let ce_names: Vec<String> = (1..=r).map(|j| format!("_ce{j}")).collect();
    e.set_mat("alpha", f.short_alpha().with_names(names_eq.clone(), ce_names.clone()));
    let mut by = Matrix::zeros(k, r);
    for i in 0..k {
        for j in 0..r {
            by[(i, j)] = f.beta[(i, j)];
        }
    }
    let _ = mat_from_rows;
    e.set_mat("pi", f.pi().with_names(names_eq.clone(), f.endog.clone()));
    let om = cross(&f.resid, f.t() as f64);
    e.set_mat("omega", om.with_names(names_eq.clone(), names_eq.clone()));
    let g = f.gammas();
    let mut gm = Matrix::zeros(k, k * (f.p - 1));
    for (l, gl) in g.iter().enumerate() {
        for i in 0..k {
            for j in 0..k {
                gm[(i, l * k + j)] = gl[(i, j)];
            }
        }
    }
    e.set_mat("Gamma", gm);
    let (c0, c1) = f.det_terms();
    e.set_mat("det", Matrix::from_rows(&(0..k).map(|i| vec![c0[i], c1[i]]).collect::<Vec<_>>()));
    let _ = by;
    let mut mask = vec![false; n];
    for &rw in &f.data.rows {
        mask[rw] = true;
    }
    e.sample = Some(mask);
}

/// Rebuild the VEC fit stored in e().
pub fn fit_from_e(ds: &DataSet, st: &Store, e: &Results) -> Result<VecFit, String> {
    if e.macros.get("cmd").map(String::as_str) != Some("vec") {
        return Err(rc_err(301, "last estimates not found; vec results required"));
    }
    let endog: Vec<String> = e.macros.get("endog").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let p = e.get("n_lags").unwrap_or(2.0) as usize;
    let r = e.get("k_ce").unwrap_or(1.0) as usize;
    let trend = Trend::parse(e.macros.get("trend").map(String::as_str))?;
    let mask = e.sample.clone().ok_or_else(|| rc_err(301, "estimation sample not found"))?;
    let tsd = ts(ds, st)?;
    let d = build(ds, st, &tsd, &endog, p, &mask)?;
    fit(d, &endog, p, r, trend)
}

pub struct Vec_;

impl CommandHandler for Vec_ {
    fn name(&self) -> &'static str {
        "vec"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let level = util::opt_num(cmd, "level", 3)?.unwrap_or(st.level);
        let tsd = ts(ds, st)?;
        if cmd.main.trim().is_empty() {
            let e = st.e.clone();
            let f = fit_from_e(ds, st, &e)?;
            return Ok(display(&f, &tsd, level, cmd));
        }
        if cmd.has_opt("constraints", 2) || cmd.has_opt("bconstraints", 2) || cmd.has_opt("aconstraints", 2) || cmd.has_opt("sindicators", 2) {
            return Err(rc_err(198, "constraints and seasonal indicators not supported"));
        }
        let endog = canon(ds, &cmd.main)?;
        if endog.len() < 2 {
            return Err(rc_err(102, "too few variables specified"));
        }
        let p = util::opt_num(cmd, "lags", 3)?.unwrap_or(2.0);
        if p < 1.0 {
            return Err(rc_err(198, "lags() must be positive"));
        }
        let r = util::opt_num(cmd, "rank", 1)?.unwrap_or(1.0) as usize;
        let trend = Trend::parse(cmd.opt("trend", 2))?;
        let mask = util::touse(ds, st, cmd)?;
        let d = build(ds, st, &tsd, &endog, p as usize, &mask)?;
        let f = fit(d, &endog, p as usize, r, trend)?;
        post(st, &f, &tsd, &cmd.raw, ds.n_obs);
        Ok(display(&f, &tsd, level, cmd))
    }
}

// ---------------------------------------------------------------------------
// vecrank
// ---------------------------------------------------------------------------

/// Osterwald-Lenum (1992) critical values by trend specification, indexed by
/// K - r (1..=11): (trace 5%, trace 1%, max 5%, max 1%).
fn ol_table(trend: Trend) -> [[f64; 11]; 4] {
    match trend {
        Trend::None => [
            [3.76, 12.53, 24.31, 39.89, 59.46, 82.49, 109.99, 141.20, 175.77, 212.67, 254.05],
            [6.65, 16.31, 29.75, 45.58, 66.52, 90.45, 119.80, 152.32, 187.31, 226.40, 269.22],
            [3.76, 11.44, 17.89, 23.80, 30.04, 36.36, 41.51, 47.99, 53.69, 59.06, 65.30],
            [6.65, 15.69, 22.99, 28.82, 35.17, 41.00, 47.15, 53.90, 59.78, 65.21, 72.36],
        ],
        Trend::RConstant => [
            [9.24, 19.96, 34.91, 53.12, 76.07, 102.14, 131.70, 165.58, 202.92, 244.15, 286.19],
            [12.97, 24.60, 41.07, 60.16, 84.45, 111.01, 143.09, 177.20, 215.74, 257.68, 307.64],
            [9.24, 15.67, 22.00, 28.14, 34.40, 40.30, 46.45, 52.00, 57.42, 63.57, 69.74],
            [12.97, 20.20, 26.81, 33.24, 39.79, 46.82, 51.91, 57.95, 63.71, 69.94, 76.63],
        ],
        Trend::Constant => [
            [3.76, 15.41, 29.68, 47.21, 68.52, 94.15, 124.24, 156.00, 192.89, 233.13, 277.71],
            [6.65, 20.04, 35.65, 54.46, 76.07, 103.18, 133.57, 168.36, 204.95, 247.18, 293.44],
            [3.76, 14.07, 20.97, 27.07, 33.46, 39.37, 45.28, 51.42, 57.12, 62.81, 68.83],
            [6.65, 18.63, 25.52, 32.24, 38.77, 45.10, 51.57, 57.95, 62.80, 69.09, 75.95],
        ],
        Trend::RTrend => [
            [12.25, 25.32, 42.44, 62.99, 87.31, 114.90, 146.76, 182.82, 222.21, 263.42, 310.81],
            [16.26, 30.45, 48.45, 70.05, 96.58, 124.75, 158.49, 196.08, 234.41, 279.07, 327.45],
            [12.25, 18.96, 25.54, 31.46, 37.52, 43.97, 49.42, 55.50, 61.29, 66.23, 72.72],
            [16.26, 23.65, 30.34, 36.65, 42.36, 49.51, 54.71, 62.46, 67.88, 73.73, 79.23],
        ],
        Trend::Trend => [
            [3.74, 18.17, 34.55, 54.64, 77.74, 104.94, 136.61, 170.80, 208.97, 250.84, 295.99],
            [6.40, 23.46, 40.49, 61.24, 85.78, 114.36, 146.99, 182.51, 222.46, 263.94, 312.58],
            [3.74, 16.87, 23.78, 30.33, 36.41, 42.48, 48.45, 54.25, 60.29, 66.10, 71.86],
            [6.40, 21.47, 28.83, 35.68, 41.58, 48.17, 54.48, 60.81, 66.91, 72.31, 78.80],
        ],
    }
}

fn crit(trend: Trend, which: usize, km_r: usize) -> f64 {
    if km_r == 0 || km_r > 11 {
        return SYSMISS;
    }
    ol_table(trend)[which][km_r - 1]
}

pub struct Vecrank;

impl CommandHandler for Vecrank {
    fn name(&self) -> &'static str {
        "vecrank"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let endog = canon(ds, &cmd.main)?;
        if endog.len() < 2 {
            return Err(rc_err(102, "too few variables specified"));
        }
        let k = endog.len();
        let p = util::opt_num(cmd, "lags", 3)?.unwrap_or(2.0) as usize;
        if p < 1 {
            return Err(rc_err(198, "lags() must be positive"));
        }
        let trend = Trend::parse(cmd.opt("trend", 2))?;
        let tsd = ts(ds, st)?;
        let mask = util::touse(ds, st, cmd)?;
        let d = build(ds, st, &tsd, &endog, p, &mask)?;
        let joh = johansen(&d, trend)?;
        let t = joh.t as f64;
        let lam = joh.lambda.clone();
        let ll: Vec<f64> = (0..=k).map(|r| joh.ll(r)).collect();
        let np: Vec<usize> = (0..=k).map(|r| nparms(k, p, trend, r)).collect();
        let trace: Vec<f64> = (0..k).map(|r| -t * lam[r..k].iter().map(|l| (1.0 - l).ln()).sum::<f64>()).collect();
        let maxs: Vec<f64> = (0..k).map(|r| -t * (1.0 - lam[r]).ln()).collect();
        let ics: Vec<(f64, f64, f64)> = (0..=k)
            .map(|r| {
                let n = np[r] as f64;
                (-2.0 * ll[r] / t + t.ln() / t * n, -2.0 * ll[r] / t + 2.0 * t.ln().ln() / t * n, -2.0 * ll[r] / t + 2.0 * n / t)
            })
            .collect();
        let level99 = cmd.has_opt("level99", 7);
        let levela = cmd.has_opt("levela", 6);
        let pick = |which: usize| (0..k).find(|&r| trace[r] < crit(trend, which, k - r)).unwrap_or(k);
        let k95 = pick(0);
        let k99 = pick(1);
        let sel_trace = if level99 { k99 } else { k95 };
        // the full-rank model is not a candidate
        let argmin = |f: &dyn Fn(usize) -> f64| (0..k).min_by(|&a, &b| f(a).total_cmp(&f(b))).unwrap();
        let ksb = argmin(&|r| ics[r].0);
        let khq = argmin(&|r| ics[r].1);
                let mut out = vec![String::new(), format!("{:^79}", "Johansen tests for cointegration").trim_end().to_string()];
        out.push(format!("{:<56}Number of obs = {:>7}", format!("Trend: {}", trend.title()), joh.t));
        out.push(format!("{:<55}Number of lags = {:>7}", format!("Sample: {} thru {}", tsd.fmt(d.times[0]), tsd.fmt(*d.times.last().unwrap())), p));
        let bar = "-".repeat(79);
        let star = |c: bool| if c { "*" } else { " " };
        let lead = |r: usize| format!("{:>7}{:>10}{:>12}{:>13}", r, np[r], format!("{:.4}", ll[r]), if r == 0 { ".".to_string() } else { format!("{:.5}", lam[r - 1]) });
        if !cmd.has_opt("notrace", 5) {
            out.push(bar.clone());
            out.push(format!("{:>66}", "Critical"));
            out.push(format!("{:<7}{:>44}{:>15}{}", "Maximum", "Trace", "value", if levela { "      Critical" } else { "" }));
            out.push(format!("{:>7}{:>10}{:>12}{:>13}{:>13}{:>15}{}", "rank", "Params", "LL", "Eigenvalue", "statistic", if levela { "5%" } else if level99 { "1%" } else { "5%" }, if levela { "        1%" } else { "" }));
            for r in 0..=k {
                if r < k {
                    let sel = if levela { r == k95 } else { r == sel_trace };
                    let cv = if levela {
                        format!("{:>15}{:>10}", fmt_cv(crit(trend, 0, k - r)), fmt_cv(crit(trend, 1, k - r)))
                    } else {
                        format!("{:>15}", fmt_cv(crit(trend, if level99 { 1 } else { 0 }, k - r)))
                    };
                    out.push(format!("{}{:>12}{}{}", lead(r), format!("{:.4}", trace[r]), star(sel), cv));
                } else {
                    out.push(lead(r));
                }
            }
            out.push(bar.clone());
            out.push("* selected rank".into());
        }
        if cmd.has_opt("max", 3) {
            out.push(bar.clone());
            out.push(format!("{:<7}{:>44}{:>17}", "Maximum", "Eigenvalue", "Critical value"));
            out.push(format!("{:>7}{:>10}{:>12}{:>13}{:>13}{:>15}{}", "rank", "Params", "LL", "", "Maximum", if levela { "5%" } else if level99 { "1%" } else { "5%" }, if levela { "        1%" } else { "" }));
            for r in 0..=k {
                if r < k {
                    let cv = if levela {
                        format!("{:>15}{:>10}", fmt_cv(crit(trend, 2, k - r)), fmt_cv(crit(trend, 3, k - r)))
                    } else {
                        format!("{:>15}", fmt_cv(crit(trend, if level99 { 3 } else { 2 }, k - r)))
                    };
                    let lt = if r == 0 { format!("{:>7}{:>10}{:>12}{:>13}", r, np[r], format!("{:.4}", ll[r]), "") } else { lead(r) };
                    out.push(format!("{}{:>12} {}", lt, format!("{:.4}", maxs[r]), cv));
                } else {
                    out.push(lead(r));
                }
            }
            out.push(bar.clone());
        }
        if cmd.has_opt("ic", 2) {
            out.push(bar.clone());
            out.push("Maximum".to_string());
            out.push(format!("{:>7}{:>10}{:>12}{:>13}{:>12} {:>11} {:>11}", "rank", "Params", "LL", "Eigenvalue", "SBIC", "HQIC", "AIC"));
            for r in 0..=k {
                out.push(format!("{}{:>12}{}{:>11}{}{:>11}{}", lead(r), format!("{:.5}", ics[r].0), star(r == ksb), format!("{:.5}", ics[r].1), star(r == khq), format!("{:.5}", ics[r].2), star(false)));
            }
            out.push(bar.clone());
            out.push("* selected rank".into());
        }
        let e = st.begin_e();
        e.set_str("cmd", "vecrank");
        e.set_str("cmdline", cmd.raw.trim());
        e.set_str("trend", trend.name());
        e.set_str("tsfmt", tsd.tfmt.clone());
        e.set("N", t);
        e.set("k_eq", k as f64);
        e.set("k_dv", k as f64);
        e.set("tmin", d.times[0]);
        e.set("tmax", *d.times.last().unwrap());
        e.set("n_lags", p as f64);
        e.set("k_ce95", k95 as f64);
        e.set("k_ce99", k99 as f64);
        e.set("k_cesbic", ksb as f64);
        e.set("k_cehqic", khq as f64);
        let rv = |v: Vec<f64>| Matrix::row_vector(&v);
        e.set_mat("max", rv(maxs));
        e.set_mat("trace", rv(trace));
        e.set_mat("ll", rv(ll));
        e.set_mat("lambda", rv(lam));
        e.set_mat("k_rank", rv(np.iter().map(|x| *x as f64).collect()));
        e.set_mat("sbic", rv(ics.iter().map(|x| x.0).collect()));
        e.set_mat("hqic", rv(ics.iter().map(|x| x.1).collect()));
        e.set_mat("aic", rv(ics.iter().map(|x| x.2).collect()));
        Ok(out.join("\n"))
    }
}

fn fmt_cv(v: f64) -> String {
    if is_missing(v) { ".".into() } else { format!("{v:.2}") }
}

// ---------------------------------------------------------------------------
// veclmar, vecnorm, vecstable
// ---------------------------------------------------------------------------

pub struct Veclmar;

impl CommandHandler for Veclmar {
    fn name(&self) -> &'static str {
        "veclmar"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let e = tsvar::est_results(st, cmd)?;
        let f = fit_from_e(ds, st, &e)?;
        let mlag = util::opt_num(cmd, "mlag", 2)?.unwrap_or(2.0).max(1.0) as usize;
        let res = tsvar::lm_autocorr(&f.data.dy, &f.x, &f.resid, mlag);
        Ok(tsvar::lm_table(&res, st))
    }
}

pub struct Vecnorm;

impl CommandHandler for Vecnorm {
    fn name(&self) -> &'static str {
        "vecnorm"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let e = tsvar::est_results(st, cmd)?;
        let f = fit_from_e(ds, st, &e)?;
        let eqs: Vec<String> = f.endog.iter().map(|x| format!("D_{}", eqname(x))).collect();
        let om = cross(&f.resid, f.t() as f64);
        Ok(tsvar::normality(&f.resid, &om, &eqs, false, cmd, st))
    }
}

pub struct Vecstable;

impl CommandHandler for Vecstable {
    fn name(&self) -> &'static str {
        "vecstable"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let e = tsvar::est_results(st, cmd)?;
        let rep = rep_from_e(&e)?;
        let comp = tsvar::companion(&rep.a);
        let k = rep.endog.len();
        tsvar::stability(st, cmd, &comp, Some(k - rep.rank))
    }
}

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![("vec", Box::new(Vec_)), ("vecrank", Box::new(Vecrank)), ("veclmar", Box::new(Veclmar)), ("vecnorm", Box::new(Vecnorm)), ("vecstable", Box::new(Vecstable))]
}
