//! Vector autoregressions: `var` (equation-by-equation OLS, which is the ML
//! and SUR estimator when every equation has the same regressors), `svar`
//! (short-run structural VAR by maximum likelihood) and the VAR diagnostics
//! `varsoc`, `varstable`, `varlmar`, `varnorm`, `vargranger` and `varwle`.
//!
//! Results are posted in e() like the other estimators; postestimation
//! commands rebuild the model from e(endog), e(lags), e(exog), the option
//! macros and e(sample) (svar stores them with a `_var` suffix).

use std::collections::HashMap;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::parser::split_top_level_words;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::{rc_err, Env};
use crate::format::{fmt_e, fmt_g};
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::tsla::{self, chol, inv, kron, mul, sym_ginv, tr};
use crate::util;

// ---------------------------------------------------------------------------
// Time-series data
// ---------------------------------------------------------------------------

/// The tsset time variable with a lookup from time value to observation.
pub struct Ts {
    pub time: Vec<f64>,
    index: HashMap<i64, usize>,
    pub tvar: String,
    pub tfmt: String,
}

pub fn ts(ds: &DataSet, st: &Store) -> Result<Ts, String> {
    // datasets saved after tsset carry the time variable in _dta[tis]
    let saved = ds.characteristics.get("_dta").and_then(|c| c.get("tis")).filter(|t| !t.is_empty() && ds.var_index.contains_key(t.as_str())).map(|t| (None, t.clone()));
    let (panel, tvar) = st.tsset.clone().or(saved).ok_or_else(|| rc_err(111, "time variable not set, use tsset varname ..."))?;
    if panel.is_some() || tvar.is_empty() {
        return Err(rc_err(459, "time-series data required; data are xtset as panel data"));
    }
    let time = util::col(ds, &tvar)?;
    let mut index = HashMap::new();
    for (i, t) in time.iter().enumerate() {
        if !is_missing(*t) {
            index.entry(t.round() as i64).or_insert(i);
        }
    }
    let tfmt = ds.get_var(&tvar).map(|v| v.format.clone()).unwrap_or_default();
    Ok(Ts { time, index, tvar, tfmt })
}

impl Ts {
    /// Value of `col` `l` periods before observation `row` (negative = lead).
    pub fn lag(&self, col: &[f64], row: usize, l: i64) -> f64 {
        if l == 0 {
            return col[row];
        }
        let t = self.time[row];
        if is_missing(t) {
            return SYSMISS;
        }
        match self.index.get(&(t.round() as i64 - l)) {
            Some(&j) => col[j],
            None => SYSMISS,
        }
    }
    pub fn row_at(&self, t: f64) -> Option<usize> {
        self.index.get(&(t.round() as i64)).copied()
    }
    pub fn fmt(&self, t: f64) -> String {
        if self.tfmt.starts_with("%t") {
            crate::format::fmt_trim(t, &self.tfmt)
        } else {
            fmt_g(t, 9)
        }
    }
}

/// Values of a variable or time-series expression (`D.x`, `L2.y`) over all
/// observations.
pub fn series(ds: &DataSet, st: &Store, name: &str) -> Result<Vec<f64>, String> {
    if ds.var_index.contains_key(name) {
        return util::col(ds, name);
    }
    let e = ferrostat_parser::parser::parse_expression(name).map_err(|e| rc_err(198, e.message))?;
    let env = Env::with_store(ds, st);
    (0..ds.n_obs).map(|i| env.eval(&e, i)?.num()).collect()
}

/// Canonical operator strings of an operator prefix: `l` → ["L"],
/// `L(0/2)` → ["", "L", "L2"], `ld` → ["LD"], `D1` → ["D"].
fn parse_op(op: &str) -> Result<Vec<String>, String> {
    let bad = || rc_err(198, format!("{op}: invalid time-series operator"));
    let chars: Vec<char> = op.chars().collect();
    let mut out = vec![String::new()];
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i].to_ascii_uppercase();
        if !"LFDS".contains(c) {
            return Err(bad());
        }
        i += 1;
        let ns: Vec<i64> = if i < chars.len() && chars[i] == '(' {
            let end = chars[i..].iter().position(|&x| x == ')').ok_or_else(bad)? + i;
            let inner: String = chars[i + 1..end].iter().collect();
            i = end + 1;
            crate::margins::numlist(&inner)?.into_iter().map(|x| x as i64).collect()
        } else {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            let s: String = chars[start..i].iter().collect();
            vec![if s.is_empty() { 1 } else { s.parse().map_err(|_| bad())? }]
        };
        let mut next = Vec::new();
        for r in &out {
            for &n in &ns {
                let piece = match n {
                    0 => String::new(),
                    1 => c.to_string(),
                    n => format!("{c}{n}"),
                };
                next.push(format!("{r}{piece}"));
            }
        }
        out = next;
    }
    Ok(out)
}

/// Expand a time-series varlist into canonical names (`d.x` → `D.x`).
pub fn canon(ds: &DataSet, text: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for w in split_top_level_words(text) {
        match w.rsplit_once('.') {
            Some((op, v)) if !op.is_empty() && op.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) => {
                let ops = parse_op(op)?;
                let vars = crate::varlist::expand(ds, v)?;
                for o in &ops {
                    for var in &vars {
                        out.push(if o.is_empty() { var.clone() } else { format!("{o}.{var}") });
                    }
                }
            }
            _ => out.extend(crate::varlist::expand(ds, &w)?),
        }
    }
    Ok(out)
}

/// Equation name of a dependent variable: `D.x` → `D_x`.
pub fn eqname(name: &str) -> String {
    name.replace('.', "_")
}

/// Coefficient name of lag `l` of `name`: `x` → `L2.x`, `D.x` → `LD.x`.
pub fn lagname(name: &str, l: usize) -> String {
    let lp = if l == 1 { "L".to_string() } else { format!("L{l}") };
    match name.rsplit_once('.') {
        Some((o, v)) => format!("{lp}{o}.{v}"),
        None => format!("{lp}.{name}"),
    }
}

// ---------------------------------------------------------------------------
// Model specification and fit
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct Spec {
    pub endog: Vec<String>,
    pub lags: Vec<usize>,
    pub exog: Vec<String>,
    pub cons: bool,
    pub dfk: bool,
    pub small: bool,
    pub lutstats: bool,
}

impl Spec {
    pub fn maxlag(&self) -> usize {
        self.lags.iter().copied().max().unwrap_or(0)
    }
    pub fn regnames(&self) -> Vec<String> {
        let mut n = Vec::new();
        for e in &self.endog {
            for &l in &self.lags {
                n.push(lagname(e, l));
            }
        }
        n.extend(self.exog.iter().cloned());
        if self.cons {
            n.push("_cons".into());
        }
        n
    }
    pub fn eqnames(&self) -> Vec<String> {
        self.endog.iter().map(|e| eqname(e)).collect()
    }
}

/// Estimation sample: rows (in time order), dependent variables and regressors.
pub struct Data {
    pub rows: Vec<usize>,
    pub times: Vec<f64>,
    pub y: Vec<Vec<f64>>,
    pub x: Vec<Vec<f64>>,
}

/// Build the sample; `need_lag` requires lags 1..=need_lag of the dependent
/// variables to exist (common sample of `varsoc`).
pub fn build(ds: &DataSet, st: &Store, tsd: &Ts, spec: &Spec, mask: &[bool], need_lag: usize) -> Result<Data, String> {
    let ycols: Vec<Vec<f64>> = spec.endog.iter().map(|e| series(ds, st, e)).collect::<Result<_, _>>()?;
    let xcols: Vec<Vec<f64>> = spec.exog.iter().map(|e| series(ds, st, e)).collect::<Result<_, _>>()?;
    let mut rows: Vec<usize> = (0..ds.n_obs).filter(|&i| mask.get(i).copied().unwrap_or(false) && !is_missing(tsd.time[i])).collect();
    rows.sort_by(|&a, &b| tsd.time[a].total_cmp(&tsd.time[b]));
    let mut d = Data { rows: Vec::new(), times: Vec::new(), y: Vec::new(), x: Vec::new() };
    'row: for &r in &rows {
        let y: Vec<f64> = ycols.iter().map(|c| c[r]).collect();
        if y.iter().any(|v| is_missing(*v)) {
            continue;
        }
        for c in &ycols {
            for l in 1..=need_lag {
                if is_missing(tsd.lag(c, r, l as i64)) {
                    continue 'row;
                }
            }
        }
        let mut x = Vec::new();
        for c in &ycols {
            for &l in &spec.lags {
                x.push(tsd.lag(c, r, l as i64));
            }
        }
        for c in &xcols {
            x.push(c[r]);
        }
        if spec.cons {
            x.push(1.0);
        }
        if x.iter().any(|v| is_missing(*v)) {
            continue;
        }
        d.rows.push(r);
        d.times.push(tsd.time[r]);
        d.y.push(y);
        d.x.push(x);
    }
    if d.rows.is_empty() {
        return Err(rc_err(2000, "no observations"));
    }
    Ok(d)
}

pub struct Fit {
    pub spec: Spec,
    pub data: Data,
    pub k: usize,
    pub m: usize,
    pub t: usize,
    /// coefficients: K equations × m regressors
    pub coef: Vec<Vec<f64>>,
    pub xtxi: Matrix,
    pub resid: Vec<Vec<f64>>,
    pub sigma_ml: Matrix,
    pub sigma: Matrix,
}

/// OLS of each equation on the common regressors.
pub fn ols_system(y: &[Vec<f64>], x: &[Vec<f64>]) -> (Vec<Vec<f64>>, Matrix, Vec<Vec<f64>>) {
    let t = y.len();
    let k = y.first().map_or(0, |r| r.len());
    let m = x.first().map_or(0, |r| r.len());
    let mut xtx = Matrix::zeros(m, m);
    let mut xty = Matrix::zeros(m, k);
    for i in 0..t {
        for a in 0..m {
            let xa = x[i][a];
            for b in 0..m {
                xtx[(a, b)] += xa * x[i][b];
            }
            for j in 0..k {
                xty[(a, j)] += xa * y[i][j];
            }
        }
    }
    let xtxi = sym_ginv(&xtx);
    let b = mul(&xtxi, &xty);
    let coef: Vec<Vec<f64>> = (0..k).map(|j| (0..m).map(|a| b[(a, j)]).collect()).collect();
    let resid: Vec<Vec<f64>> = (0..t).map(|i| (0..k).map(|j| y[i][j] - (0..m).map(|a| x[i][a] * coef[j][a]).sum::<f64>()).collect()).collect();
    (coef, xtxi, resid)
}

pub fn cross(u: &[Vec<f64>], div: f64) -> Matrix {
    let k = u.first().map_or(0, |r| r.len());
    let mut s = Matrix::zeros(k, k);
    for r in u {
        for i in 0..k {
            for j in 0..k {
                s[(i, j)] += r[i] * r[j];
            }
        }
    }
    s.scale(1.0 / div)
}

pub fn fit_data(spec: &Spec, data: Data) -> Result<Fit, String> {
    let t = data.rows.len();
    let k = spec.endog.len();
    let m = data.x.first().map_or(0, |r| r.len());
    if t <= m {
        return Err(rc_err(2001, "insufficient observations"));
    }
    let (coef, xtxi, resid) = ols_system(&data.y, &data.x);
    let sigma_ml = cross(&resid, t as f64);
    let sigma = if spec.dfk { cross(&resid, (t - m) as f64) } else { sigma_ml.clone() };
    Ok(Fit { spec: spec.clone(), data, k, m, t, coef, xtxi, resid, sigma_ml, sigma })
}

impl Fit {
    pub fn ll(&self) -> f64 {
        -(self.t as f64) / 2.0 * (self.k as f64 * ((2.0 * std::f64::consts::PI).ln() + 1.0) + self.sigma_ml.det().ln())
    }
    /// (aic, hqic, sbic, fpe)
    pub fn ic(&self, lutstats: bool) -> (f64, f64, f64, f64) {
        let t = self.t as f64;
        let k = self.k as f64;
        let m = self.m as f64;
        let det = self.sigma_ml.det();
        let fpe = det * ((t + m) / (t - m)).powf(k);
        if lutstats {
            let pk2 = self.spec.lags.len() as f64 * k * k;
            let ld = det.ln();
            (ld + 2.0 * pk2 / t, ld + 2.0 * t.ln().ln() / t * pk2, ld + t.ln() / t * pk2, fpe)
        } else {
            let ll = self.ll();
            let tp = k * m;
            (-2.0 * ll / t + 2.0 * tp / t, -2.0 * ll / t + 2.0 * t.ln().ln() / t * tp, -2.0 * ll / t + t.ln() / t * tp, fpe)
        }
    }
    /// e(V): Σ ⊗ (X'X)⁻¹ in equation-major order.
    pub fn vce(&self) -> Matrix {
        kron(&self.sigma, &self.xtxi)
    }
    pub fn b(&self) -> Vec<f64> {
        self.coef.iter().flatten().copied().collect()
    }
    /// Lag coefficient matrices A_1..A_p (p = largest lag).
    pub fn a_mats(&self) -> Vec<Matrix> {
        a_mats_from(&self.spec, &self.b(), self.m)
    }
    pub fn df_r(&self) -> f64 {
        (self.t - self.m) as f64
    }
}

/// Lag matrices from a coefficient vector laid out as e(b) of `var`.
pub fn a_mats_from(spec: &Spec, b: &[f64], m: usize) -> Vec<Matrix> {
    let k = spec.endog.len();
    let nl = spec.lags.len();
    let p = spec.maxlag();
    let mut a = vec![Matrix::zeros(k, k); p];
    for i in 0..k {
        for j in 0..k {
            for (li, &l) in spec.lags.iter().enumerate() {
                a[l - 1][(i, j)] = b[i * m + j * nl + li];
            }
        }
    }
    a
}

/// Exogenous coefficient columns (K × R) in e(b) layout.
pub fn exog_coefs(spec: &Spec, b: &[f64], m: usize) -> Matrix {
    let k = spec.endog.len();
    let off = k * spec.lags.len();
    let r = spec.exog.len();
    let mut out = Matrix::zeros(k, r);
    for i in 0..k {
        for j in 0..r {
            out[(i, j)] = b[i * m + off + j];
        }
    }
    out
}

/// Companion matrix of lag matrices A_1..A_p.
pub fn companion(a: &[Matrix]) -> Matrix {
    let p = a.len();
    let k = a.first().map_or(0, |m| m.rows);
    let mut c = Matrix::zeros(k * p, k * p);
    for (l, al) in a.iter().enumerate() {
        for i in 0..k {
            for j in 0..k {
                c[(i, l * k + j)] = al[(i, j)];
            }
        }
    }
    for i in k..k * p {
        c[(i, i - k)] = 1.0;
    }
    c
}

// ---------------------------------------------------------------------------
// Options and e() round trip
// ---------------------------------------------------------------------------

fn parse_lags(cmd: &Command, default: &[usize]) -> Result<Vec<usize>, String> {
    match cmd.opt("lags", 3) {
        Some(t) => {
            let mut l: Vec<usize> = crate::margins::numlist(t)?
                .into_iter()
                .map(|x| if x >= 1.0 && x.fract() == 0.0 { Ok(x as usize) } else { Err(rc_err(198, "lags() must contain positive integers")) })
                .collect::<Result<_, _>>()?;
            l.sort();
            l.dedup();
            Ok(l)
        }
        None => Ok(default.to_vec()),
    }
}

pub fn spec_from_cmd(ds: &DataSet, cmd: &Command, default_lags: &[usize]) -> Result<Spec, String> {
    let endog = canon(ds, &cmd.main)?;
    if endog.is_empty() {
        return Err(rc_err(102, "too few variables specified"));
    }
    if cmd.has_opt("constraints", 2) {
        return Err(rc_err(198, "constraints() not supported"));
    }
    Ok(Spec {
        endog,
        lags: parse_lags(cmd, default_lags)?,
        exog: cmd.opt("exog", 3).map(|t| canon(ds, t)).transpose()?.unwrap_or_default(),
        cons: !cmd.has_opt("noconstant", 5),
        dfk: cmd.has_opt("dfk", 3),
        small: cmd.has_opt("small", 3),
        lutstats: cmd.has_opt("lutstats", 3),
    })
}

/// The e() results a postestimation command works on: `estimates(name)` or
/// the active ones.
pub fn est_results(st: &Store, cmd: &Command) -> Result<Results, String> {
    match cmd.opt("estimates", 3) {
        Some(n) => st.estimates.get(n.trim()).cloned().ok_or_else(|| rc_err(111, format!("estimation result {} not found", n.trim()))),
        None => Ok(st.e.clone()),
    }
}

fn mac(e: &Results, name: &str) -> String {
    e.macros.get(name).or_else(|| e.macros.get(&format!("{name}_var"))).cloned().unwrap_or_default()
}

/// The VAR specification stored by `var` or `svar`.
pub fn spec_from_e(e: &Results) -> Result<Spec, String> {
    let cmd = e.macros.get("cmd").cloned().unwrap_or_default();
    if cmd != "var" && cmd != "svar" {
        return Err(rc_err(301, "last estimates not found; var or svar results required"));
    }
    let words = |s: String| s.split_whitespace().map(String::from).collect::<Vec<_>>();
    Ok(Spec {
        endog: words(mac(e, "endog")),
        lags: mac(e, "lags").split_whitespace().filter_map(|x| x.parse().ok()).collect(),
        exog: words(mac(e, "exog")),
        cons: mac(e, "nocons").is_empty(),
        dfk: !mac(e, "dfk").is_empty(),
        small: !mac(e, "small").is_empty(),
        lutstats: !mac(e, "lutstats").is_empty(),
    })
}

/// Refit the VAR stored in `e` on its estimation sample.
pub fn fit_from_e(ds: &DataSet, st: &Store, e: &Results) -> Result<Fit, String> {
    let spec = spec_from_e(e)?;
    let mut mask = e.sample.clone().ok_or_else(|| rc_err(301, "estimation sample not found"))?;
    if mask.len() > ds.n_obs {
        return Err(rc_err(459, "data have changed since estimation"));
    }
    // observations appended after estimation (e.g. forecast rows) are outside the sample
    mask.resize(ds.n_obs, false);
    let tsd = ts(ds, st)?;
    let data = build(ds, st, &tsd, &spec, &mask, 0)?;
    fit_data(&spec, data)
}

fn sample_mask(n: usize, rows: &[usize]) -> Vec<bool> {
    let mut m = vec![false; n];
    for &r in rows {
        m[r] = true;
    }
    m
}

fn n_gaps(times: &[f64]) -> usize {
    times.windows(2).filter(|w| w[1] - w[0] > 1.0 + 1e-9).count()
}

struct EqStat {
    parms: usize,
    rmse: f64,
    r2: f64,
    chi2: f64,
    df: f64,
    p: f64,
    f: f64,
    pf: f64,
    ll: f64,
}

fn eq_stats(f: &Fit) -> Vec<EqStat> {
    let t = f.t as f64;
    let names = f.spec.regnames();
    let free: Vec<usize> = (0..f.m).filter(|&j| names[j] != "_cons").collect();
    (0..f.k)
        .map(|i| {
            let y: Vec<f64> = f.data.y.iter().map(|r| r[i]).collect();
            let ssr: f64 = f.resid.iter().map(|r| r[i] * r[i]).sum();
            let mean = if f.spec.cons { util::mean(&y) } else { 0.0 };
            let sst: f64 = y.iter().map(|v| (v - mean).powi(2)).sum();
            let rmse = f.sigma[(i, i)].sqrt();
            // Wald test of the non-constant coefficients
            let q = free.len();
            let mut vs = Matrix::zeros(q, q);
            for (a, &ja) in free.iter().enumerate() {
                for (b, &jb) in free.iter().enumerate() {
                    vs[(a, b)] = f.sigma[(i, i)] * f.xtxi[(ja, jb)];
                }
            }
            let vi = sym_ginv(&vs);
            let bs: Vec<f64> = free.iter().map(|&j| f.coef[i][j]).collect();
            let chi2: f64 = (0..q).map(|a| (0..q).map(|b| bs[a] * vi[(a, b)] * bs[b]).sum::<f64>()).sum();
            let df = q as f64;
            EqStat {
                parms: f.m,
                rmse,
                r2: 1.0 - ssr / sst,
                chi2,
                df,
                p: dist::chi2tail(df, chi2),
                f: chi2 / df,
                pf: dist::ftail(df, f.df_r(), chi2 / df),
                ll: -t / 2.0 * ((2.0 * std::f64::consts::PI).ln() + 1.0 + (ssr / t).ln()),
            }
        })
        .collect()
}

/// Post the e() results of a VAR fit (`suffix` = "_var" for svar).
fn post_var(e: &mut Results, f: &Fit, ds_n: usize, tsd: &Ts, raw: &str, suffix: &str) {
    let spec = &f.spec;
    let eqn = spec.eqnames();
    let regs = spec.regnames();
    let names: Vec<String> = eqn.iter().flat_map(|q| regs.iter().map(move |r| format!("{q}:{r}"))).collect();
    let stats = eq_stats(f);
    let (aic, hqic, sbic, fpe) = f.ic(spec.lutstats);
    let s = |n: &str| format!("{n}{suffix}");
    let t = f.t as f64;
    e.set(&s("N"), t);
    e.set(&s("N_gaps"), n_gaps(&f.data.times) as f64);
    e.set(&s("k"), (f.k * f.m) as f64);
    e.set(&s("k_eq"), f.k as f64);
    e.set(&s("k_dv"), f.k as f64);
    e.set(&s("df_eq"), f.m as f64);
    e.set(&s("df_m"), (f.m - spec.cons as usize) as f64);
    if spec.small {
        e.set(&s("df_r"), f.df_r());
    }
    e.set(&s("ll"), f.ll());
    if spec.dfk {
        let lld = -t / 2.0 * (f.k as f64 * ((2.0 * std::f64::consts::PI).ln() + 1.0) + f.sigma.det().ln());
        e.set(&s("ll_dfk"), lld);
    }
    for (i, q) in stats.iter().enumerate() {
        let n = i + 1;
        e.set(&s(&format!("obs_{n}")), t);
        e.set(&s(&format!("k_{n}")), q.parms as f64);
        e.set(&s(&format!("df_m{n}")), q.df);
        e.set(&s(&format!("r2_{n}")), q.r2);
        e.set(&s(&format!("ll_{n}")), q.ll);
        e.set(&s(&format!("chi2_{n}")), q.chi2);
        e.set(&s(&format!("rmse_{n}")), q.rmse);
        if spec.small {
            e.set(&s(&format!("df_r{n}")), f.df_r());
            e.set(&s(&format!("F_{n}")), q.f);
            e.set(&s(&format!("p_{n}")), q.pf);
        } else {
            e.set(&s(&format!("p_{n}")), q.p);
        }
    }
    e.set(&s("aic"), aic);
    e.set(&s("hqic"), hqic);
    e.set(&s("sbic"), sbic);
    e.set(&s("fpe"), fpe);
    e.set(&s("mlag"), spec.maxlag() as f64);
    e.set("tmin", f.data.times[0]);
    e.set("tmax", *f.data.times.last().unwrap());
    e.set(&s("detsig"), f.sigma.det());
    e.set(&s("detsig_ml"), f.sigma_ml.det());
    if suffix.is_empty() {
        e.set("rank", (f.k * f.m) as f64);
        e.set_str("cmdline", raw.trim());
        e.set_str("title", "Vector autoregression");
        e.set_str("properties", "b V");
        e.set_str("predict", "var_p");
        e.set_str("vce", "ols");
    }
    e.set_str(&s("depvar"), spec.endog.join(" "));
    e.set_str(&s("endog"), spec.endog.join(" "));
    if !spec.exog.is_empty() {
        e.set_str(&s("exog"), spec.exog.join(" "));
        let mut base: Vec<String> = spec.exog.iter().map(|x| x.rsplit('.').next().unwrap().to_string()).collect();
        base.dedup();
        e.set_str(&s("exogvars"), base.join(" "));
    }
    e.set_str(&s("eqnames"), eqn.join(" "));
    e.set_str(&s("lags"), spec.lags.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(" "));
    if !spec.cons {
        e.set_str(&s("nocons"), "nocons");
    }
    if spec.dfk {
        e.set_str(&s("dfk"), "dfk");
    }
    if spec.small {
        e.set_str("small", "small");
    }
    if spec.lutstats {
        e.set_str(&s("lutstats"), "lutstats");
    }
    e.set_str("timevar", tsd.tvar.clone());
    e.set_str("tsfmt", tsd.tfmt.clone());
    let b = f.b();
    let bm = Matrix::row_vector(&b).with_names(vec!["y1".into()], names.clone());
    let vm = f.vce().with_names(names.clone(), names);
    let sig = f.sigma.clone().with_names(spec.endog.clone(), spec.endog.clone());
    if suffix.is_empty() {
        e.set_mat("b", bm.clone());
        e.set_mat("V", vm);
        e.set_mat("bf", bm);
        e.set_mat("Sigma", sig);
    } else {
        e.set_mat("b_var", bm.clone());
        e.set_mat("bf_var", bm);
        e.set_mat("V_var", vm);
        e.set_mat("Sigma", sig);
    }
    e.sample = Some(sample_mask(ds_n, &f.data.rows));
}

/// Header and tables of `var`.
fn display_var(f: &Fit, tsd: &Ts, level: f64) -> String {
    let spec = &f.spec;
    let (aic, hqic, sbic, fpe) = f.ic(spec.lutstats);
    let mut out = vec![String::new(), "Vector autoregression".to_string(), String::new()];
    let line = |l: String, k: &str, v: String| format!("{l:<48}{k:<17} = {v:>10}");
    out.push(line(format!("Sample: {} thru {}", tsd.fmt(f.data.times[0]), tsd.fmt(*f.data.times.last().unwrap())), "Number of obs", f.t.to_string()));
    out.push(line(format!("Log likelihood = {:>9}{}", fmt_g(f.ll(), 8), if spec.lutstats { "  (lutstats)" } else { "" }), "AIC", fmt_g(aic, 8)));
    out.push(line(format!("FPE            = {:>9}", fmt_e(fpe, 2)), "HQIC", fmt_g(hqic, 8)));
    out.push(line(format!("Det(Sigma_ml)  = {:>9}", fmt_e(f.sigma_ml.det(), 2)), "SBIC", fmt_g(sbic, 8)));
    out.push(String::new());
    let small = spec.small;
    out.push(format!("{:<15}{:>9}{:>10}{:>9}{:>10}{:>11}", "Equation", "Parms", "RMSE", "R-sq", if small { "F" } else { "chi2" }, if small { "P > F" } else { "P>chi2" }));
    out.push("-".repeat(64));
    for (i, q) in eq_stats(f).iter().enumerate() {
        let name = crate::summarize::abbrev(&eqname(&spec.endog[i]), 15);
        let (st, p) = if small { (q.f, q.pf) } else { (q.chi2, q.p) };
        out.push(format!("{:<15}{:>8}{:>12}{:>9.4}{:>11}{:>9.4}", name, q.parms, fmt_g(q.rmse, 7), q.r2, fmt_g(st, 8), p));
    }
    out.push("-".repeat(64));
    out.push(String::new());
    let eqn = spec.eqnames();
    let regs = spec.regnames();
    let names: Vec<String> = eqn.iter().flat_map(|q| regs.iter().map(move |r| format!("{q}:{r}"))).collect();
    let b = f.b();
    let v = f.vce();
    let no = vec![false; names.len()];
    let tspec = crate::est::TableSpec {
        depvar: "",
        names: &names,
        b: &b,
        v: &v,
        omitted: &no,
        is_base: &no,
        df: if small { Some(f.df_r()) } else { None },
        level,
        eform: None,
        coef_title: "Coefficient",
    };
    out.extend(crate::est::coef_table(&tspec));
    if spec.dfk {
        out.push("Note: Small-sample degrees-of-freedom adjustment applied when estimating".into());
        out.push("      covariance matrix of residuals.".into());
    }
    out.join("\n")
}

fn level_of(cmd: &Command, st: &Store) -> Result<f64, String> {
    Ok(util::opt_num(cmd, "level", 3)?.unwrap_or(st.level))
}

/// `var depvarlist [if] [in] [, lags() exog() noconstant dfk small lutstats]`
pub struct Var;

impl CommandHandler for Var {
    fn name(&self) -> &'static str {
        "var"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let level = level_of(cmd, st)?;
        if cmd.main.trim().is_empty() {
            // replay
            let e = st.e.clone();
            if e.macros.get("cmd").map(String::as_str) != Some("var") {
                return Err(rc_err(301, "last estimates not found"));
            }
            let f = fit_from_e(ds, st, &e)?;
            let tsd = ts(ds, st)?;
            return Ok(display_var(&f, &tsd, level));
        }
        let spec = spec_from_cmd(ds, cmd, &[1, 2])?;
        let tsd = ts(ds, st)?;
        let mask = util::touse(ds, st, cmd)?;
        let data = build(ds, st, &tsd, &spec, &mask, 0)?;
        let f = fit_data(&spec, data)?;
        let e = st.begin_e();
        e.set_str("cmd", "var");
        post_var(e, &f, ds.n_obs, &tsd, &cmd.raw, "");
        Ok(display_var(&f, &tsd, level))
    }
}

// ---------------------------------------------------------------------------
// Boxed tables
// ---------------------------------------------------------------------------

/// A table framed by `+---+` borders with a first column separated by `|`.
fn boxed(head: (String, String), rows: &[Option<Vec<String>>], w0: usize) -> Vec<String> {
    let width = rows.iter().flatten().map(|r| r[1].chars().count()).max().unwrap_or(0).max(head.1.chars().count());
    let mut out = Vec::new();
    let total = w0 + 2 + width;
    out.push(format!("  +{}+", "-".repeat(total)));
    out.push(format!("  |{:<w0$} |{:<width$}|", head.0, head.1));
    out.push(format!("  |{}+{}|", "-".repeat(w0 + 1), "-".repeat(width)));
    for r in rows {
        match r {
            Some(r) => out.push(format!("  |{:<w0$} |{:<width$}|", r[0], r[1])),
            None => out.push(format!("  |{}+{}|", "-".repeat(w0 + 1), "-".repeat(width))),
        }
    }
    out.push(format!("  +{}+", "-".repeat(total)));
    out
}

// ---------------------------------------------------------------------------
// varsoc
// ---------------------------------------------------------------------------

pub struct Varsoc;

impl CommandHandler for Varsoc {
    fn name(&self) -> &'static str {
        "varsoc"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let level = level_of(cmd, st)?;
        let tsd = ts(ds, st)?;
        let (mut spec, maxlag, mask) = if cmd.main.trim().is_empty() {
            let e = est_results(st, cmd)?;
            let spec = spec_from_e(&e)?;
            let ml = spec.maxlag();
            let mask = e.sample.clone().ok_or_else(|| rc_err(301, "estimation sample not found"))?;
            (spec, ml, mask)
        } else {
            let mut spec = spec_from_cmd(ds, cmd, &[])?;
            spec.lags.clear();
            let ml = util::opt_num(cmd, "maxlag", 4)?.unwrap_or(4.0);
            if ml < 1.0 {
                return Err(rc_err(198, "maxlag() must be positive"));
            }
            (spec, ml as usize, util::touse(ds, st, cmd)?)
        };
        if cmd.has_opt("lutstats", 3) {
            spec.lutstats = true;
        }
        let k = spec.endog.len() as f64;
        let mut fits = Vec::new();
        for p in 0..=maxlag {
            let mut s = spec.clone();
            s.lags = (1..=p).collect();
            s.dfk = false;
            let data = build(ds, st, &tsd, &s, &mask, maxlag)?;
            fits.push(fit_data(&s, data)?);
        }
        let n = fits[0].t;
        let ll: Vec<f64> = fits.iter().map(|f| f.ll()).collect();
        let ics: Vec<(f64, f64, f64, f64)> = fits.iter().map(|f| f.ic(spec.lutstats)).collect();
        let lr: Vec<f64> = (0..=maxlag).map(|p| if p == 0 { SYSMISS } else { 2.0 * (ll[p] - ll[p - 1]) }).collect();
        let df = k * k;
        let pv: Vec<f64> = lr.iter().map(|&x| if is_missing(x) { SYSMISS } else { dist::chi2tail(df, x) }).collect();
        let alpha = 1.0 - level / 100.0;
        let lr_star = (1..=maxlag).rev().find(|&p| pv[p] < alpha);
        let argmin = |f: &dyn Fn(usize) -> f64| (0..=maxlag).min_by(|&a, &b| f(a).total_cmp(&f(b))).unwrap();
        let fpe_s = argmin(&|p| ics[p].3);
        let aic_s = argmin(&|p| ics[p].0);
        let hq_s = argmin(&|p| ics[p].1);
        let sb_s = argmin(&|p| ics[p].2);
        let star = |c: bool| if c { "*" } else { " " };
        let sig = |v: f64, d: usize| fmt_g6(v, d);
        let mut out = vec![String::new()];
        out.push(format!("   {}", if spec.lutstats { "Lutkepohl's lag-order selection criteria" } else { "Lag-order selection criteria" }));
        out.push(String::new());
        let f0 = &fits[0];
        let left = format!("   Sample: {} thru {}", tsd.fmt(f0.data.times[0]), tsd.fmt(*f0.data.times.last().unwrap()));
        out.push(format!("{left:<62}Number of obs = {n}"));
        let head = (" Lag".to_string(), format!("{:>9}{:>8} {:>5}{:>7}{:>9} {:>9} {:>9} {:>9}  ", "LL", "LR", "df", "p", "FPE", "AIC", "HQIC", "SBIC"));
        let mut rows = Vec::new();
        let sep: usize = util::opt_num(cmd, "separator", 3)?.unwrap_or(0.0) as usize;
        for p in 0..=maxlag {
            if sep > 0 && p > 0 && p % sep == 0 {
                rows.push(None);
            }
            let (lrs, dfs, ps) = if p == 0 {
                (format!("{:>9}", ""), String::from("     "), String::from("       "))
            } else {
                (format!("{:>8}{}", sig(lr[p], 5), star(lr_star == Some(p))), format!("{:>5}", df as i64), format!("{:>7.3}", pv[p]))
            };
            let body = format!(
                "{:>9}{}{}{}{:>9}{}{:>9}{}{:>9}{}{:>9}{} ",
                sig(ll[p], 6),
                lrs,
                dfs,
                ps,
                fmt_e(ics[p].3, 1),
                star(fpe_s == p),
                sig(ics[p].0, 6),
                star(aic_s == p),
                sig(ics[p].1, 6),
                star(hq_s == p),
                sig(ics[p].2, 6),
                star(sb_s == p)
            );
            rows.push(Some(vec![format!("{p:>4}"), body]));
        }
        out.extend(boxed(head, &rows, 4));
        out.push("   * optimal lag".into());
        out.push(format!("   Endogenous: {}", spec.endog.join(" ")));
        let mut ex = spec.exog.clone();
        if spec.cons {
            ex.push("_cons".into());
        }
        out.push(format!("    Exogenous: {}", ex.join(" ")));
        let r = st.begin_r();
        r.set("N", n as f64);
        r.set("tmin", f0.data.times[0]);
        r.set("tmax", *f0.data.times.last().unwrap());
        r.set("mlag", maxlag as f64);
        r.set("N_gaps", n_gaps(&f0.data.times) as f64);
        r.set_str("endog", spec.endog.join(" "));
        if !spec.exog.is_empty() {
            r.set_str("exog", spec.exog.join(" "));
        }
        if spec.lutstats {
            r.set_str("lutstats", "lutstats");
        }
        let mut m = Matrix::zeros(maxlag + 1, 9);
        for p in 0..=maxlag {
            let row = [p as f64, ll[p], lr[p], if p == 0 { SYSMISS } else { df }, pv[p], ics[p].3, ics[p].0, ics[p].1, ics[p].2];
            for (j, v) in row.iter().enumerate() {
                m[(p, j)] = *v;
            }
        }
        let cn = ["lag", "ll", "lr", "df", "p", "fpe", "aic", "hqic", "sbic"].iter().map(|s| s.to_string()).collect();
        r.set_mat("stats", m.with_names((0..=maxlag).map(|p| format!("r{}", p + 1)).collect(), cn));
        Ok(out.join("\n"))
    }
}

/// `%w.0g` with at most `d` significant digits.
pub fn fmt_g6(v: f64, d: usize) -> String {
    if is_missing(v) {
        return ".".into();
    }
    if v == 0.0 {
        return "0".into();
    }
    let exp = v.abs().log10().floor() as i32;
    if exp >= -4 && exp < d as i32 {
        let dec = (d as i32 - 1 - exp).max(0) as usize;
        let s = format!("{:.*}", dec, v);
        let s = if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s };
        return s.replacen("0.", ".", if s.starts_with("0.") || s.starts_with("-0.") { 1 } else { 0 });
    }
    fmt_e(v, d.saturating_sub(1))
}

// ---------------------------------------------------------------------------
// varstable
// ---------------------------------------------------------------------------

/// Eigenvalue table (and optional graph) shared by varstable and vecstable.
pub fn stability(st: &mut Store, cmd: &Command, comp: &Matrix, vec_unit: Option<usize>) -> Result<String, String> {
    let ev = tsla::eig_general(comp);
    let mut out = vec![String::new(), "   Eigenvalue stability condition".to_string()];
    let mut rows = Vec::new();
    for &(re, im) in &ev {
        let ztext = if im.abs() > 1e-12 * (1.0 + re.abs()) {
            format!("{:>10} {} {:>9}i", fmt_g(re, 8), if im < 0.0 { "-" } else { "+" }, fmt_g(im.abs(), 8))
        } else {
            format!("{:>10}", fmt_g(re, 8))
        };
        let modulus = (re * re + im * im).sqrt();
        rows.push(Some(vec![format!(" {ztext:<25}"), format!("{:>10}   ", fmt_g(modulus, 7))]));
    }
    out.extend(boxed((format!("{:^26}", "Eigenvalue"), format!("{:^13}", "Modulus")), &rows, 26));
    match vec_unit {
        Some(u) => {
            out.push(if u == 1 { "   The VECM specification imposes a unit modulus.".to_string() } else { format!("   The VECM specification imposes {u} unit moduli.") });
        }
        None => {
            if ev.iter().all(|&(r, i)| (r * r + i * i).sqrt() < 1.0) {
                out.push("   All the eigenvalues lie inside the unit circle.".into());
                out.push("   VAR satisfies stability condition.".into());
            } else {
                out.push("   At least one eigenvalue is at least 1.0.".into());
                out.push("   VAR does not satisfy stability condition.".into());
            }
        }
    }
    if let Some(name) = cmd.opt("amat", 2) {
        st.matrices.insert(name.trim().to_string(), comp.clone());
    }
    let r = st.begin_r();
    let n = ev.len();
    let cn: Vec<String> = (1..=n).map(|i| format!("c{i}")).collect();
    r.set_mat("Re", Matrix::row_vector(&ev.iter().map(|e| e.0).collect::<Vec<_>>()).with_names(vec!["r1".into()], cn.clone()));
    r.set_mat("Im", Matrix::row_vector(&ev.iter().map(|e| e.1).collect::<Vec<_>>()).with_names(vec!["r1".into()], cn.clone()));
    r.set_mat("Modulus", Matrix::row_vector(&ev.iter().map(|e| (e.0 * e.0 + e.1 * e.1).sqrt()).collect::<Vec<_>>()).with_names(vec!["r1".into()], cn));
    if cmd.has_opt("graph", 2) {
        stability_graph(st, cmd, &ev)?;
    }
    Ok(out.join("\n"))
}

fn stability_graph(st: &mut Store, cmd: &Command, ev: &[(f64, f64)]) -> Result<(), String> {
    use crate::graph::{Graph, Layer, LayerKind};
    let mut g = Graph::default();
    let circle = |r: f64| -> Vec<(f64, f64)> { (0..=200).map(|i| { let a = i as f64 / 200.0 * std::f64::consts::TAU; (r * a.cos(), r * a.sin()) }).collect() };
    if !cmd.has_opt("nogrid", 3) {
        for i in 1..=9 {
            g.layers.push(Layer { kind: LayerKind::Line, points: circle(i as f64 / 10.0), color: Some((220, 220, 220)), ..Default::default() });
        }
    }
    g.layers.push(Layer { kind: LayerKind::Line, points: circle(1.0), color: Some((90, 90, 90)), ..Default::default() });
    let mut pts = Layer { kind: LayerKind::Scatter, points: ev.iter().map(|e| (e.0, e.1)).collect(), ..Default::default() };
    if cmd.has_opt("dlabel", 3) {
        pts.mlabels = ev.iter().map(|e| format!("{:.2}", 1.0 - (e.0 * e.0 + e.1 * e.1).sqrt())).collect();
    } else if cmd.has_opt("modlabel", 3) {
        pts.mlabels = ev.iter().map(|e| format!("{:.2}", (e.0 * e.0 + e.1 * e.1).sqrt())).collect();
    }
    g.layers.push(pts);
    g.title = "Roots of the companion matrix".into();
    g.xtitle = "Real".into();
    g.ytitle = "Imaginary".into();
    g.legend = false;
    crate::graph::finish(&mut g, cmd, st)?;
    Ok(())
}

pub struct Varstable;

impl CommandHandler for Varstable {
    fn name(&self) -> &'static str {
        "varstable"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let e = est_results(st, cmd)?;
        let spec = spec_from_e(&e)?;
        let b = e.matrices.get("b_var").or_else(|| e.matrices.get("b")).cloned().ok_or_else(|| rc_err(301, "e(b) not found"))?;
        let m = spec.regnames().len();
        let _ = ds;
        let comp = companion(&a_mats_from(&spec, &b.data, m));
        stability(st, cmd, &comp, None)
    }
}

// ---------------------------------------------------------------------------
// varlmar
// ---------------------------------------------------------------------------

/// LM statistics for residual autocorrelation at lags 1..=mlag given the
/// regressors and residuals of a VAR (Davidson-MacKinnon zero padding).
pub fn lm_autocorr(y: &[Vec<f64>], x: &[Vec<f64>], u: &[Vec<f64>], mlag: usize) -> Vec<(f64, f64, f64)> {
    let t = y.len();
    let k = y.first().map_or(0, |r| r.len());
    let det0 = cross(u, t as f64).det();
    (1..=mlag)
        .map(|s| {
            let xa: Vec<Vec<f64>> = (0..t)
                .map(|i| {
                    let mut r = x[i].clone();
                    for j in 0..k {
                        r.push(if i >= s { u[i - s][j] } else { 0.0 });
                    }
                    r
                })
                .collect();
            let (_, _, ua) = ols_system(y, &xa);
            let det1 = cross(&ua, t as f64).det();
            let d = xa[0].len() as f64;
            let lm = (t as f64 - d - 0.5) * (det0 / det1).ln();
            let df = (k * k) as f64;
            (lm, df, dist::chi2tail(df, lm))
        })
        .collect()
}

pub fn lm_table(res: &[(f64, f64, f64)], st: &mut Store) -> String {
    let mut out = vec![String::new(), "   Lagrange-multiplier test".to_string()];
    let rows: Vec<Option<Vec<String>>> = res.iter().enumerate().map(|(i, r)| Some(vec![format!("{:>4}", i + 1), format!("{:>10.4}{:>8}{:>13}    ", r.0, r.1 as i64, format!("{:.5}", r.2))])).collect();
    out.extend(boxed((" lag".into(), format!("{:>10}{:>8}{:>13}    ", "chi2", "df", "Prob > chi2")), &rows, 4));
    out.push("   H0: no autocorrelation at lag order".into());
    let mut m = Matrix::zeros(res.len(), 3);
    for (i, r) in res.iter().enumerate() {
        m[(i, 0)] = r.0;
        m[(i, 1)] = r.1;
        m[(i, 2)] = r.2;
    }
    let r = st.begin_r();
    r.set_mat("lm", m.with_names((1..=res.len()).map(|i| format!("r{i}")).collect(), vec!["chi2".into(), "df".into(), "p".into()]));
    out.join("\n")
}

pub struct Varlmar;

impl CommandHandler for Varlmar {
    fn name(&self) -> &'static str {
        "varlmar"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let e = est_results(st, cmd)?;
        let f = fit_from_e(ds, st, &e)?;
        let mlag = util::opt_num(cmd, "mlag", 2)?.unwrap_or(2.0).max(1.0) as usize;
        let res = lm_autocorr(&f.data.y, &f.data.x, &f.resid, mlag);
        Ok(lm_table(&res, st))
    }
}

// ---------------------------------------------------------------------------
// varnorm
// ---------------------------------------------------------------------------

/// Normality statistics of orthogonalised residuals: per equation
/// (skewness, kurtosis) and the chi2 statistics.
pub fn normality(u: &[Vec<f64>], sigma: &Matrix, eqs: &[String], dfk: bool, cmd: &Command, st: &mut Store) -> String {
    let t = u.len() as f64;
    let k = sigma.rows;
    let p = chol(sigma);
    let pinv = inv(&p).unwrap_or_else(|| Matrix::identity(k));
    let w: Vec<Vec<f64>> = u.iter().map(|r| (0..k).map(|i| (0..k).map(|j| pinv[(i, j)] * r[j]).sum()).collect()).collect();
    let b1: Vec<f64> = (0..k).map(|i| w.iter().map(|r| r[i].powi(3)).sum::<f64>() / t).collect();
    let b2: Vec<f64> = (0..k).map(|i| w.iter().map(|r| r[i].powi(4)).sum::<f64>() / t).collect();
    let l1: Vec<f64> = b1.iter().map(|b| t * b * b / 6.0).collect();
    let l2: Vec<f64> = b2.iter().map(|b| t * (b - 3.0).powi(2) / 24.0).collect();
    let jb: Vec<f64> = (0..k).map(|i| l1[i] + l2[i]).collect();
    let all = !cmd.has_opt("jbera", 2) && !cmd.has_opt("skewness", 2) && !cmd.has_opt("kurtosis", 2);
    let mut out = vec![String::new()];
    let mut rmats: Vec<(&str, Matrix)> = Vec::new();
    let mut table = |title: &str, stat: Option<(&str, &[f64])>, chi: &[f64], df1: f64, out: &mut Vec<String>, key: &'static str| {
        out.push(format!("   {title}"));
        let tot: f64 = chi.iter().sum();
        let mut rows: Vec<Option<Vec<String>>> = Vec::new();
        let mut m = Matrix::zeros(k + 1, 3);
        for i in 0..=k {
            let (name, c, df) = if i < k { (eqs[i].clone(), chi[i], df1) } else { ("ALL".to_string(), tot, df1 * k as f64) };
            let pv = dist::chi2tail(df, c);
            m[(i, 0)] = c;
            m[(i, 1)] = df;
            m[(i, 2)] = pv;
            let st_txt = match stat {
                Some((_, v)) if i < k => format!("{:>11}", fmt_g6(v[i], 5)),
                Some(_) => format!("{:>11}", ""),
                None => String::new(),
            };
            rows.push(Some(vec![format!("{:>19}", crate::summarize::abbrev(&name, 19)), format!("{}{:>10.3}{:>5}{:>13.5}   ", st_txt, c, df as i64, pv)]));
        }
        let hb = match stat {
            Some((h, _)) => format!("{:>11}{:>10}{:>5}{:>13}   ", h, "chi2", "df", "Prob > chi2"),
            None => format!("{:>10}{:>5}{:>13}   ", "chi2", "df", "Prob > chi2"),
        };
        out.extend(boxed((format!("{:>19}", "Equation"), hb), &rows, 19));
        let mut names: Vec<String> = eqs.to_vec();
        names.push("ALL".into());
        rmats.push((key, m.with_names(names, vec!["chi2".into(), "df".into(), "p".into()])));
        out.push(String::new());
    };
    if all || cmd.has_opt("jbera", 2) {
        table("Jarque-Bera test", None, &jb, 2.0, &mut out, "jb");
    }
    if all || cmd.has_opt("skewness", 2) {
        table("Skewness test", Some(("Skewness", &b1)), &l1, 1.0, &mut out, "skewness");
    }
    if all || cmd.has_opt("kurtosis", 2) {
        table("Kurtosis test", Some(("Kurtosis", &b2)), &l2, 1.0, &mut out, "kurtosis");
    }
    if dfk {
        out.push("   dfk estimator used in computations".into());
    }
    let r = st.begin_r();
    for (k, m) in rmats {
        r.set_mat(k, m);
    }
    if dfk {
        r.set_str("dfk", "dfk");
    }
    out.join("\n").trim_end().to_string()
}

pub struct Varnorm;

impl CommandHandler for Varnorm {
    fn name(&self) -> &'static str {
        "varnorm"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let e = est_results(st, cmd)?;
        let f = fit_from_e(ds, st, &e)?;
        let eqs = f.spec.eqnames();
        Ok(normality(&f.resid, &f.sigma, &eqs, f.spec.dfk, cmd, st))
    }
}

// ---------------------------------------------------------------------------
// vargranger, varwle
// ---------------------------------------------------------------------------

fn wald(f: &Fit, idx: &[usize]) -> f64 {
    let b = f.b();
    let v = f.vce();
    let q = idx.len();
    let mut vs = Matrix::zeros(q, q);
    for (a, &i) in idx.iter().enumerate() {
        for (c, &j) in idx.iter().enumerate() {
            vs[(a, c)] = v[(i, j)];
        }
    }
    let vi = sym_ginv(&vs);
    (0..q).map(|a| (0..q).map(|c| b[idx[a]] * vi[(a, c)] * b[idx[c]]).sum::<f64>()).sum()
}

pub struct Vargranger;

impl CommandHandler for Vargranger {
    fn name(&self) -> &'static str {
        "vargranger"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let e = est_results(st, cmd)?;
        let f = fit_from_e(ds, st, &e)?;
        let k = f.k;
        let nl = f.spec.lags.len();
        let eqs = f.spec.eqnames();
        let small = f.spec.small;
        let mut rows: Vec<Option<Vec<String>>> = Vec::new();
        let mut stats: Vec<Vec<f64>> = Vec::new();
        let sep = util::opt_num(cmd, "separator", 3)?.unwrap_or(0.0) as usize;
        let mut count = 0;
        for i in 0..k {
            let mut tests: Vec<(String, Vec<usize>)> = Vec::new();
            for j in (0..k).filter(|&j| j != i) {
                tests.push((eqs[j].clone(), (0..nl).map(|l| i * f.m + j * nl + l).collect()));
            }
            if k > 2 {
                tests.push(("ALL".into(), (0..k).filter(|&j| j != i).flat_map(|j| (0..nl).map(move |l| i * f.m + j * nl + l)).collect()));
            }
            if i > 0 {
                rows.push(None);
            }
            for (name, idx) in tests {
                if sep > 0 && count > 0 && count % sep == 0 && !matches!(rows.last(), Some(None)) {
                    rows.push(None);
                }
                count += 1;
                let w = wald(&f, &idx);
                let df = idx.len() as f64;
                let body = if small {
                    let fv = w / df;
                    let p = dist::ftail(df, f.df_r(), fv);
                    stats.push(vec![fv, df, f.df_r(), p]);
                    format!("{:>9}{:>5}{:>7}{:>11.4}   ", fmt_g(fv, 6), df as i64, f.df_r() as i64, p)
                } else {
                    let p = dist::chi2tail(df, w);
                    stats.push(vec![w, df, p]);
                    format!("{:>9}{:>5}{:>14.4}   ", fmt_g(w, 6), df as i64, p)
                };
                rows.push(Some(vec![format!("{:>18}{:>19}", crate::summarize::abbrev(&eqs[i], 18), crate::summarize::abbrev(&name, 18)), body]));
            }
        }
        let mut out = vec![String::new(), "   Granger causality Wald tests".to_string()];
        let hb = if small { format!("{:>9}{:>5}{:>7}{:>11}   ", "F", "df", "df_r", "Prob > F") } else { format!("{:>9}{:>5}{:>14}   ", "chi2", "df", "Prob > chi2") };
        out.extend(boxed((format!("{:>18}{:>19}", "Equation", "Excluded"), hb), &rows, 37));
        let ncol = stats.first().map_or(3, |r| r.len());
        let mut m = Matrix::zeros(stats.len(), ncol);
        for (i, r) in stats.iter().enumerate() {
            for (j, v) in r.iter().enumerate() {
                m[(i, j)] = *v;
            }
        }
        let cn: Vec<String> = if small { vec!["F", "df", "df_r", "p"] } else { vec!["chi2", "df", "p"] }.into_iter().map(String::from).collect();
        let rn = (1..=stats.len()).map(|i| format!("r{i}")).collect();
        st.begin_r().set_mat("gstats", m.with_names(rn, cn));
        Ok(out.join("\n"))
    }
}

pub struct Varwle;

impl CommandHandler for Varwle {
    fn name(&self) -> &'static str {
        "varwle"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let e = est_results(st, cmd)?;
        let f = fit_from_e(ds, st, &e)?;
        let k = f.k;
        let nl = f.spec.lags.len();
        let eqs = f.spec.eqnames();
        let small = f.spec.small;
        let mut out = Vec::new();
        let mut mats: Vec<Vec<Vec<f64>>> = vec![Vec::new(); 4];
        for eq in 0..=k {
            let title = if eq < k { eqs[eq].clone() } else { "All".into() };
            out.push(String::new());
            out.push(format!("   Equation: {title}"));
            let mut rows = Vec::new();
            let mut row_stats = vec![Vec::new(); 4];
            for (li, &l) in f.spec.lags.iter().enumerate() {
                let eqs_used: Vec<usize> = if eq < k { vec![eq] } else { (0..k).collect() };
                let idx: Vec<usize> = eqs_used.iter().flat_map(|&i| (0..k).map(move |j| i * f.m + j * nl + li)).collect();
                let w = wald(&f, &idx);
                let df = idx.len() as f64;
                let body = if small {
                    let fv = w / df;
                    let p = dist::ftail(df, f.df_r(), fv);
                    row_stats[0].push(fv);
                    row_stats[1].push(df);
                    row_stats[2].push(f.df_r());
                    row_stats[3].push(p);
                    format!("{:>11}{:>8}{:>8}{:>10.4}   ", fmt_g6(fv, 6), df as i64, f.df_r() as i64, p)
                } else {
                    let p = dist::chi2tail(df, w);
                    row_stats[0].push(w);
                    row_stats[1].push(df);
                    row_stats[3].push(p);
                    format!("{:>11}{:>8}{:>13.3}   ", fmt_g(w, 8), df as i64, p)
                };
                rows.push(Some(vec![format!("{l:>4}"), body]));
            }
            let hb = if small { format!("{:>11}{:>8}{:>8}{:>10}   ", "F", "df", "df_r", "Prob > F") } else { format!("{:>11}{:>8}{:>13}   ", "chi2", "df", "Prob > chi2") };
            out.extend(boxed((" lag".into(), hb), &rows, 4));
            for c in 0..4 {
                mats[c].push(row_stats[c].clone());
            }
        }
        let r = st.begin_r();
        let to_m = |rows: &Vec<Vec<f64>>| {
            let mut m = Matrix::zeros(rows.len(), rows.first().map_or(0, |r| r.len()));
            for (i, r) in rows.iter().enumerate() {
                for (j, v) in r.iter().enumerate() {
                    m[(i, j)] = *v;
                }
            }
            let mut rn = eqs.clone();
            rn.push("All".into());
            let cols = m.cols;
            m.with_names(rn, f.spec.lags.iter().take(cols).map(|l| format!("lag{l}")).collect())
        };
        if small {
            r.set_mat("F", to_m(&mats[0]));
            r.set_mat("df", to_m(&mats[1]));
            r.set_mat("df_r", to_m(&mats[2]));
        } else {
            r.set_mat("chi2", to_m(&mats[0]));
            r.set_mat("df", to_m(&mats[1]));
        }
        r.set_mat("p", to_m(&mats[3]));
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// svar (short-run)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum Cell {
    Fixed(f64),
    Free(usize),
}

/// Cells of A or B (column-major) from aeq()/acns() style matrices.
fn cells_from(st: &Store, cmd: &Command, eq: &str, cns: &str, k: usize, default: &dyn Fn(usize, usize) -> Cell, next: &mut usize) -> Result<Vec<Cell>, String> {
    let get = |name: &str| -> Result<Option<Matrix>, String> {
        match cmd.opt(name, 4) {
            Some(n) => {
                let m = st.matrices.get(n.trim()).cloned().ok_or_else(|| rc_err(111, format!("matrix {} not found", n.trim())))?;
                if m.rows != k || m.cols != k {
                    return Err(rc_err(503, format!("{name}() matrix must be {k} x {k}")));
                }
                Ok(Some(m))
            }
            None => Ok(None),
        }
    };
    let meq = get(eq)?;
    let mcns = get(cns)?;
    let mut cells = Vec::with_capacity(k * k);
    let mut groups: HashMap<i64, usize> = HashMap::new();
    for j in 0..k {
        for i in 0..k {
            let c = if let Some(m) = &meq {
                let v = m[(i, j)];
                if is_missing(v) {
                    *next += 1;
                    Cell::Free(*next - 1)
                } else {
                    Cell::Fixed(v)
                }
            } else if let Some(m) = &mcns {
                let v = m[(i, j)];
                if is_missing(v) {
                    *next += 1;
                    Cell::Free(*next - 1)
                } else {
                    let id = *groups.entry(v.round() as i64).or_insert_with(|| {
                        *next += 1;
                        *next - 1
                    });
                    Cell::Free(id)
                }
            } else {
                match default(i, j) {
                    Cell::Free(_) => {
                        *next += 1;
                        Cell::Free(*next - 1)
                    }
                    f => f,
                }
            };
            cells.push(c);
        }
    }
    Ok(cells)
}

fn fill(cells: &[Cell], th: &[f64], k: usize) -> Matrix {
    let mut m = Matrix::zeros(k, k);
    for j in 0..k {
        for i in 0..k {
            m[(i, j)] = match cells[j * k + i] {
                Cell::Fixed(v) => v,
                Cell::Free(p) => th[p],
            };
        }
    }
    m
}

fn svar_ll(a: &Matrix, b: &Matrix, sigma: &Matrix, n: f64) -> f64 {
    let k = a.rows as f64;
    let Some(bi) = inv(b) else { return f64::NEG_INFINITY };
    let w = mul(&bi, a);
    let d = w.det();
    if d == 0.0 || !d.is_finite() {
        return f64::NEG_INFINITY;
    }
    let wtw = mul(&tr(&w), &w);
    let trace: f64 = (0..a.rows).map(|i| (0..a.rows).map(|j| wtw[(i, j)] * sigma[(j, i)]).sum::<f64>()).sum();
    -n * k / 2.0 * (2.0 * std::f64::consts::PI).ln() + n / 2.0 * (d * d).ln() - n / 2.0 * trace
}

/// Commutation matrix K_K: K vec(G) = vec(G').
fn commutation(k: usize) -> Matrix {
    let mut c = Matrix::zeros(k * k, k * k);
    for i in 0..k {
        for j in 0..k {
            c[(i * k + j, j * k + i)] = 1.0;
        }
    }
    c
}

/// Expected information of vec(A), vec(B).
fn svar_info(a: &Matrix, b: &Matrix, n: f64) -> Matrix {
    let k = a.rows;
    let bi = inv(b).unwrap_or_else(|| Matrix::identity(k));
    let w = mul(&bi, a);
    let wi = inv(&w).unwrap_or_else(|| Matrix::identity(k));
    let left = kron(&tr(&wi), &bi);
    let right = kron(&Matrix::identity(k), &bi).scale(-1.0);
    let mut m = Matrix::zeros(k * k, 2 * k * k);
    for i in 0..k * k {
        for j in 0..k * k {
            m[(i, j)] = left[(i, j)];
            m[(i, k * k + j)] = right[(i, j)];
        }
    }
    let ik = tsla::add(&Matrix::identity(k * k), &commutation(k));
    mul(&mul(&tr(&m), &ik), &m).scale(n)
}

pub struct Svar;

impl CommandHandler for Svar {
    fn name(&self) -> &'static str {
        "svar"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.has_opt("lreq", 3) || cmd.has_opt("lrcns", 3) {
            return Err(rc_err(198, "long-run constraints not supported"));
        }
        if cmd.has_opt("acons", 4) || cmd.has_opt("bcons", 4) || cmd.has_opt("acon", 4) || cmd.has_opt("bcon", 4) {
            return Err(rc_err(198, "constraint lists not supported; use aeq(), acns(), beq() or bcns()"));
        }
        let level = level_of(cmd, st)?;
        let spec = spec_from_cmd(ds, cmd, &[1, 2])?;
        let tsd = ts(ds, st)?;
        let mask = util::touse(ds, st, cmd)?;
        let data = build(ds, st, &tsd, &spec, &mask, 0)?;
        let f = fit_data(&spec, data)?;
        let k = f.k;
        let mut np = 0usize;
        let ca = cells_from(st, cmd, "aeq", "acns", k, &|i, j| Cell::Fixed(if i == j { 1.0 } else { 0.0 }), &mut np)?;
        let cb = cells_from(st, cmd, "beq", "bcns", k, &|i, j| if i == j { Cell::Free(0) } else { Cell::Fixed(0.0) }, &mut np)?;
        let n = f.t as f64;
        let sigma = f.sigma.clone();
        // selection matrix S: vec(A,B) = S θ + fixed
        let mut s = Matrix::zeros(2 * k * k, np);
        for (c, cell) in ca.iter().chain(cb.iter()).enumerate() {
            if let Cell::Free(p) = cell {
                s[(c, *p)] = 1.0;
            }
        }
        let ll_of = |th: &[f64]| svar_ll(&fill(&ca, th, k), &fill(&cb, th, k), &sigma, n);
        let start = vec![1.0 + np as f64 / 100.0; np];
        let mut th = start.clone();
        let mut ll = ll_of(&th);
        if !ll.is_finite() {
            // fall back to Cholesky-like starting values
            th = vec![0.5; np];
            ll = ll_of(&th);
        }
        let mut log = vec![ll];
        let mut converged = false;
        for _ in 0..300 {
            let g = tsla::jacobian(&|x: &[f64]| vec![ll_of(x)], &th);
            let info = mul(&mul(&tr(&s), &svar_info(&fill(&ca, &th, k), &fill(&cb, &th, k), n)), &s);
            let ii = sym_ginv(&info);
            let dir: Vec<f64> = (0..np).map(|i| (0..np).map(|j| ii[(i, j)] * g[(0, j)]).sum()).collect();
            let mut step = 1.0;
            let mut improved = false;
            for _ in 0..40 {
                let cand: Vec<f64> = th.iter().zip(&dir).map(|(a, d)| a + step * d).collect();
                let lc = ll_of(&cand);
                if lc.is_finite() && lc >= ll - 1e-12 * ll.abs().max(1.0) {
                    let change = lc - ll;
                    th = cand;
                    ll = lc;
                    improved = true;
                    log.push(ll);
                    if change.abs() < 1e-11 * ll.abs().max(1.0) {
                        converged = true;
                    }
                    break;
                }
                step /= 2.0;
            }
            if !improved || converged {
                if !improved {
                    converged = true;
                }
                break;
            }
        }
        if !converged {
            return Err(rc_err(430, "convergence not achieved"));
        }
        let mut amat = fill(&ca, &th, k);
        let mut bmat = fill(&cb, &th, k);
        // sign normalisation: positive diagonal of free diagonal elements
        for i in 0..k {
            let row_b_diag_free = matches!(cb[i * k + i], Cell::Free(_));
            let flip = if row_b_diag_free { bmat[(i, i)] < 0.0 } else { matches!(ca[i * k + i], Cell::Free(_)) && amat[(i, i)] < 0.0 };
            // flipping row i of W = B⁻¹A: allowed when the other cells of the
            // affected row/column are free or zero
            if flip {
                let col_b_ok = (0..k).all(|r| r == i || matches!(cb[i * k + r], Cell::Fixed(0.0)) || matches!(cb[i * k + r], Cell::Free(_)));
                let row_a_ok = (0..k).all(|c| matches!(ca[c * k + i], Cell::Fixed(0.0)) || matches!(ca[c * k + i], Cell::Free(_)));
                if row_b_diag_free && col_b_ok {
                    for r in 0..k {
                        bmat[(r, i)] = -bmat[(r, i)];
                    }
                } else if row_a_ok {
                    for c in 0..k {
                        amat[(i, c)] = -amat[(i, c)];
                    }
                }
            }
        }
        // re-read θ from the normalised matrices
        for (c, cell) in ca.iter().enumerate() {
            if let Cell::Free(p) = cell {
                th[*p] = amat[(c % k, c / k)];
            }
        }
        for (c, cell) in cb.iter().enumerate() {
            if let Cell::Free(p) = cell {
                th[*p] = bmat[(c % k, c / k)];
            }
        }
        let info = mul(&mul(&tr(&s), &svar_info(&amat, &bmat, n)), &s);
        let vth = sym_ginv(&info);
        let vfull = mul(&mul(&s, &vth), &tr(&s));
        let mut names = Vec::new();
        let mut bvec = Vec::new();
        for (mname, mm) in [("A", &amat), ("B", &bmat)] {
            for j in 0..k {
                for i in 0..k {
                    names.push(format!("/{mname}:{}_{}", i + 1, j + 1));
                    bvec.push(mm[(i, j)]);
                }
            }
        }
        let fixed: Vec<bool> = ca.iter().chain(cb.iter()).map(|c| matches!(c, Cell::Fixed(_))).collect();
        let nfree_ident = k * (k + 1) / 2;
        let rank = {
            let (_, dropped) = info.sym_inverse(1e-12);
            np - dropped.len()
        };
        let oid_df = nfree_ident as i64 - rank as i64;
        let ll_var = f.ll();
        let chi2_oid = 2.0 * (ll_var - ll);
        // output
        let mut out = vec![String::new(), "Estimating short-run parameters".to_string(), String::new()];
        out.extend(crate::est::iteration_log(&log, "Log likelihood"));
        out.push(String::new());
        out.push("Structural vector autoregression".into());
        out.push(String::new());
        let mut cn = 0;
        for (mname, cells) in [("A", &ca), ("B", &cb)] {
            for i in 0..k {
                for j in 0..k {
                    if let Cell::Fixed(v) = cells[j * k + i] {
                        cn += 1;
                        out.push(format!(" ({cn:>2})  [/{mname}]{}_{} = {}", i + 1, j + 1, fmt_g(v, 9)));
                    }
                }
            }
        }
        // equality constraints from *cns()
        let mut seen: HashMap<usize, (String, usize)> = HashMap::new();
        for (c, cell) in ca.iter().chain(cb.iter()).enumerate() {
            if let Cell::Free(p) = cell {
                let mname = if c < k * k { "A" } else { "B" };
                let cc = c % (k * k);
                let label = format!("[/{mname}]{}_{}", cc % k + 1, cc / k + 1);
                match seen.get(p) {
                    Some((first, _)) => {
                        cn += 1;
                        out.push(format!(" ({cn:>2})  {first} - {label} = 0"));
                    }
                    None => {
                        seen.insert(*p, (label, c));
                    }
                }
            }
        }
        out.push(String::new());
        let line = |l: String, kk: &str, v: String| format!("{l:<48}{kk:<17} = {v:>10}");
        out.push(line(format!("Sample: {} thru {}", tsd.fmt(f.data.times[0]), tsd.fmt(*f.data.times.last().unwrap())), "Number of obs", f.t.to_string()));
        out.push(line(if oid_df > 0 { "Overidentified model".into() } else { "Exactly identified model".into() }, "Log likelihood", fmt_g(ll, 8)));
        out.push(String::new());
        let rows: Vec<CoefRow> = (0..names.len())
            .map(|j| CoefRow { name: names[j].clone(), b: bvec[j], se: vfull[(j, j)].max(0.0).sqrt(), kind: if fixed[j] { RowKind::Constrained } else { RowKind::Normal } })
            .collect();
        out.extend(coef_rows_table(&rows, None, level, ""));
        if oid_df > 0 {
            out.push(format!("LR test of identifying restrictions: chi2({:>3}) = {:<8}  Prob > chi2 = {:.3}", oid_df, fmt_g(chi2_oid, 7), dist::chi2tail(oid_df as f64, chi2_oid)));
        }
        let saved: Vec<(&str, Matrix)> = ["aeq", "acns", "beq", "bcns"].iter().filter_map(|o| cmd.opt(o, 4).and_then(|n| st.matrices.get(n.trim()).cloned()).map(|m| (*o, m))).collect();
        let e = st.begin_e();
        e.set_str("cmd", "svar");
        post_var(e, &f, ds.n_obs, &tsd, &cmd.raw, "_var");
        e.set("N", n);
        e.set("ll", ll);
        e.set("N_cns", cn as f64);
        e.set("k_eq", 2.0);
        e.set("k_dv", k as f64);
        e.set("rank", rank as f64);
        e.set("ic_ml", log.len().saturating_sub(1) as f64);
        e.set("rc_ml", 0.0);
        if oid_df > 0 {
            e.set("chi2_oid", chi2_oid);
            e.set("oid_df", oid_df as f64);
        }
        e.set_str("cmdline", cmd.raw.trim());
        e.set_str("title", "Structural vector autoregression");
        e.set_str("properties", "b V");
        e.set_str("depvar", spec.endog.join(" "));
        let en = spec.endog.clone();
        e.set_mat("A", amat.clone().with_names(en.clone(), en.clone()));
        e.set_mat("B", bmat.clone().with_names(en.clone(), en.clone()));
        e.set_mat("b", Matrix::row_vector(&bvec).with_names(vec!["y1".into()], names.clone()));
        e.set_mat("V", vfull.with_names(names.clone(), names));
        for (o, m) in saved {
            e.set_mat(o, m);
        }
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// Coefficient table with constrained / fixed rows
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
pub enum RowKind {
    Normal,
    /// value shown with "(constrained)"
    Constrained,
    /// value without inference (dots)
    NoSe,
    /// "(omitted)"
    Omitted,
}

pub struct CoefRow {
    pub name: String,
    pub b: f64,
    pub se: f64,
    pub kind: RowKind,
}

/// Coefficient table supporting constrained and fixed parameters; names are
/// `eq:coef`. Also stashes r(table).
pub fn coef_rows_table(rows: &[CoefRow], df: Option<f64>, level: f64, title: &str) -> Vec<String> {
    let crit = match df {
        Some(d) => dist::invttail(d, (1.0 - level / 100.0) / 2.0),
        None => dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0),
    };
    let stat = if df.is_some() { "t" } else { "z" };
    let lw = 12usize;
    let mut out = Vec::new();
    out.push("-".repeat(lw + 66));
    out.push(format!("{:>lw$} | Coefficient  Std. err.  {:>6}    P>|{}|     [{}% conf. interval]", title, stat, stat, fmt_g(level, 4)));
    out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
    let mut last_eq = String::new();
    let mut last_head = String::new();
    let mut tm = Matrix::zeros(9, rows.len());
    for (j, r) in rows.iter().enumerate() {
        let (eq, nm) = r.name.split_once(':').map(|(a, b)| (a.to_string(), b.to_string())).unwrap_or((String::new(), r.name.clone()));
        if eq != last_eq {
            if !last_eq.is_empty() {
                out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(64)));
            }
            out.push(format!("{:<lw$} |", crate::summarize::abbrev(&eq, lw)));
            last_eq = eq;
            last_head.clear();
        }
        let (label, head) = crate::est::display_name(&nm);
        match head {
            Some(h) if h != last_head => {
                if !out.last().is_some_and(|l| l.trim_end().ends_with('|')) {
                    out.push(format!("{:>lw$} |", ""));
                }
                out.push(format!("{:>lw$} |", crate::summarize::abbrev(&h, lw)));
                last_head = h;
            }
            Some(_) => {}
            None => {
                if !last_head.is_empty() {
                    out.push(format!("{:>lw$} |", ""));
                    last_head.clear();
                }
            }
        }
        let lab = crate::summarize::abbrev(&label, lw);
        let col: [f64; 6];
        match r.kind {
            RowKind::Normal => {
                let z = r.b / r.se;
                let p = match df {
                    Some(d) => 2.0 * dist::ttail(d, z.abs()),
                    None => 2.0 * dist::normal(-z.abs()),
                };
                out.push(format!("{:>lw$} | {:>10}  {:>9}  {:>7.2}   {:.3}    {:>9}   {:>9}", lab, fmt_g(r.b, 8), fmt_g(r.se, 8), z, p, fmt_g(r.b - crit * r.se, 8), fmt_g(r.b + crit * r.se, 8)));
                col = [r.b, r.se, z, p, r.b - crit * r.se, r.b + crit * r.se];
            }
            RowKind::Constrained => {
                out.push(format!("{:>lw$} | {:>10}  (constrained)", lab, fmt_g(r.b, 8)));
                col = [r.b, SYSMISS, SYSMISS, SYSMISS, SYSMISS, SYSMISS];
            }
            RowKind::NoSe => {
                out.push(format!("{:>lw$} | {:>10}  {:>9}  {:>7}   {:>5}    {:>9}   {:>9}", lab, fmt_g(r.b, 8), ".", ".", ".", ".", "."));
                col = [r.b, SYSMISS, SYSMISS, SYSMISS, SYSMISS, SYSMISS];
            }
            RowKind::Omitted => {
                out.push(format!("{:>lw$} | {:>10}  (omitted)", lab, "0"));
                col = [0.0, SYSMISS, SYSMISS, SYSMISS, SYSMISS, SYSMISS];
            }
        }
        for (i, v) in col.iter().enumerate() {
            tm[(i, j)] = *v;
        }
        tm[(6, j)] = df.unwrap_or(SYSMISS);
        tm[(7, j)] = crit;
        tm[(8, j)] = 0.0;
    }
    out.push("-".repeat(lw + 66));
    let rn = ["b", "se", stat, "pvalue", "ll", "ul", "df", "crit", "eform"].iter().map(|s| s.to_string()).collect();
    crate::est::stash_table(tm.with_names(rn, rows.iter().map(|r| r.name.clone()).collect()), level);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operators() {
        assert_eq!(parse_op("l").unwrap(), vec!["L"]);
        assert_eq!(parse_op("L(0/2)").unwrap(), vec!["", "L", "L2"]);
        assert_eq!(parse_op("ld").unwrap(), vec!["LD"]);
        assert_eq!(lagname("D.x", 2), "L2D.x");
        assert_eq!(lagname("x", 1), "L.x");
        assert_eq!(fmt_g6(-24.38290, 6), "-24.3829");
        assert_eq!(fmt_g6(0.0062, 4), ".0062");
    }
}

/// Commands of the multivariate time-series family, registered by the engine.
pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    let mut v: Vec<(&'static str, Box<dyn CommandHandler>)> = vec![
        ("var", Box::new(Var)),
        ("svar", Box::new(Svar)),
        ("varsoc", Box::new(Varsoc)),
        ("varstable", Box::new(Varstable)),
        ("varlmar", Box::new(Varlmar)),
        ("varnorm", Box::new(Varnorm)),
        ("vargranger", Box::new(Vargranger)),
        ("varwle", Box::new(Varwle)),
    ];
    v.extend(crate::tsvec::handlers());
    v.extend(crate::tsirf::handlers());
    v.extend(crate::tsfcast::handlers());
    v
}
