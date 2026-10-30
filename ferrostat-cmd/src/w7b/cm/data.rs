//! Shared machinery of the choice-model family: the `cmset` declaration,
//! preparation of long-format choice data (one row per case and
//! alternative), quasi-random draws, the GHK simulator and a case-level
//! likelihood wrapper with numerical derivatives.

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::dist;
use crate::est::{self, Design, DesignSpec, Likelihood};
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::util;
use crate::xmodels as xm;

// ---------------------------------------------------------------------------
// cmset declaration (dataset characteristics, so it travels with the data)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct CmSet {
    pub case: String,
    pub alt: Option<String>,
    pub panel: Option<String>,
    pub time: Option<String>,
}

const KEYS: [&str; 4] = ["_cm_case", "_cm_altvar", "_cm_panelvar", "_cm_timevar"];

pub fn get_set(ds: &DataSet) -> Option<CmSet> {
    let c = ds.characteristics.get("_dta")?;
    let case = c.get(KEYS[0])?.clone();
    if case.is_empty() || !ds.var_index.contains_key(&case) {
        return None;
    }
    let opt = |k: &str| c.get(k).filter(|s| !s.is_empty()).cloned();
    Some(CmSet { case, alt: opt(KEYS[1]), panel: opt(KEYS[2]), time: opt(KEYS[3]) })
}

pub fn put_set(ds: &mut DataSet, s: &CmSet) {
    let c = ds.characteristics.entry("_dta".into()).or_default();
    c.insert(KEYS[0].into(), s.case.clone());
    for (k, v) in [(KEYS[1], &s.alt), (KEYS[2], &s.panel), (KEYS[3], &s.time)] {
        match v {
            Some(v) => {
                c.insert(k.into(), v.clone());
            }
            None => {
                c.remove(k);
            }
        }
    }
}

pub fn clear_set(ds: &mut DataSet) {
    if let Some(c) = ds.characteristics.get_mut("_dta") {
        for k in KEYS {
            c.remove(k);
        }
    }
}

/// The declaration in force: `case()`/`group()` and `alternatives()`
/// options override `cmset`.
pub fn resolve(ds: &DataSet, cmd: &Command) -> Result<CmSet, String> {
    let mut s = get_set(ds).unwrap_or_default();
    if let Some(c) = cmd.opt("case", 4).or_else(|| cmd.opt("group", 2)) {
        s = CmSet { case: crate::varlist::resolve_name(ds, c.trim())?, alt: None, panel: None, time: None };
    }
    if let Some(a) = cmd.opt("alternatives", 3) {
        s.alt = Some(crate::varlist::resolve_name(ds, a.trim())?);
    }
    if s.case.is_empty() {
        return Err(rc_err(459, "data not cmset; use cmset to declare the case and alternatives variables"));
    }
    Ok(s)
}

// ---------------------------------------------------------------------------
// prepared choice data
// ---------------------------------------------------------------------------

pub struct CmData {
    pub depvar: String,
    pub set: CmSet,
    /// design over the estimation rows (y = depvar)
    pub d: Design,
    /// estimation rows grouped by case (indexes into the design rows)
    pub cases: Vec<Vec<usize>>,
    /// panel index of each case (cases themselves when not panel data)
    pub case_panel: Vec<usize>,
    pub npanel: usize,
    /// distinct alternatives, the alternative of each row, labels, base
    pub alts: Vec<f64>,
    pub alt_of: Vec<usize>,
    pub alt_labels: Vec<String>,
    pub base: usize,
    /// model columns: alternative-specific variables, then case variables
    /// and constants interacted with the non-base alternatives
    pub names: Vec<String>,
    pub cols: Vec<Vec<f64>>,
    pub is_cons: Vec<bool>,
    pub n_as: usize,
    pub casevars: Vec<String>,
}

/// Which outcome the model needs: exactly one chosen alternative per
/// case, or ranks.
#[derive(Clone, Copy, PartialEq)]
pub enum Need {
    One,
    Rank,
}

/// Equation-name-safe label.
fn eq_label(s: &str) -> String {
    let t: String = s.chars().map(|c| if c.is_whitespace() || c == ':' || c == ',' { '_' } else { c }).collect();
    if t.is_empty() {
        "_".into()
    } else {
        t
    }
}

pub fn opt_vars(ds: &DataSet, cmd: &Command, name: &str, min: usize) -> Result<Vec<String>, String> {
    match cmd.opt(name, min) {
        Some(c) if !c.trim().is_empty() => crate::varlist::expand(ds, c.trim()),
        _ => Ok(vec![]),
    }
}

/// Prepare choice data from `depvar indepvars`, `casevars()`,
/// `basealternative()` and `noconstant`.
pub fn prepare(ds: &DataSet, st: &Store, cmd: &Command, need: Need) -> Result<CmData, String> {
    let set = resolve(ds, cmd)?;
    let main = cmd.main.trim();
    let (dep, indep) = main.split_once(char::is_whitespace).map(|(a, b)| (a, b.trim())).unwrap_or((main, ""));
    if dep.is_empty() {
        return Err(rc_err(100, "varlist required"));
    }
    let dep = crate::varlist::resolve_name(ds, dep)?;
    let casevars = opt_vars(ds, cmd, "casevars", 5)?;
    let nocons = cmd.has_opt("noconstant", 3);
    let mut extra = vec![set.case.clone()];
    if let Some(a) = &set.alt {
        extra.push(a.clone());
    }
    if let Some(p) = &set.panel {
        extra.push(p.clone());
    }
    extra.extend(casevars.iter().cloned());
    let mut d = est::build(ds, st, cmd, DesignSpec { depvar: Some(&dep), indep, noconstant: true, weights: "", extra })?;
    // case-wise deletion: drop whole cases with any excluded row
    let touse = util::touse(ds, st, cmd)?;
    let casec = util::col(ds, &set.case)?;
    let mut bad: std::collections::HashSet<u64> = Default::default();
    for i in 0..ds.n_obs {
        if touse[i] && !d.mask[i] && !is_missing(casec[i]) {
            bad.insert(casec[i].to_bits());
        }
    }
    if !bad.is_empty() {
        let keep: Vec<bool> = d.rows.iter().map(|&r| !bad.contains(&casec[r].to_bits())).collect();
        d = xm::subset(&d, &keep);
    }
    if d.n() == 0 {
        return Err(rc_err(2000, "no observations"));
    }
    let (_, cases) = crate::xtnl::panel_groups(ds, &set.case, &d.rows)?;
    // alternatives
    let (alts, alt_of, alt_labels) = match &set.alt {
        Some(a) => {
            let av = ds.get_var(a).ok_or_else(|| rc_err(111, format!("variable {a} not found")))?;
            if av.var_type.is_string() {
                return Err(rc_err(109, format!("alternatives variable {a} must be numeric")));
            }
            let ac = util::col(ds, a)?;
            let mut alts: Vec<f64> = d.rows.iter().map(|&r| ac[r]).collect();
            alts.sort_by(f64::total_cmp);
            alts.dedup();
            let alt_of: Vec<usize> = d.rows.iter().map(|&r| alts.binary_search_by(|k| k.total_cmp(&ac[r])).unwrap()).collect();
            let labels: Vec<String> = alts
                .iter()
                .map(|v| {
                    let t = alt_of.iter().position(|&k| alts[k] == *v).unwrap();
                    eq_label(&util::cell_text(ds, av, d.rows[t], true))
                })
                .collect();
            for g in &cases {
                let mut seen = vec![false; alts.len()];
                for &t in g {
                    if seen[alt_of[t]] {
                        return Err(rc_err(459, format!("variable {a} has replicate levels for one or more cases")));
                    }
                    seen[alt_of[t]] = true;
                }
            }
            (alts, alt_of, labels)
        }
        None => (Vec::new(), vec![usize::MAX; d.n()], Vec::new()),
    };
    // outcome checks
    match need {
        Need::One => {
            for g in &cases {
                let pos = g.iter().filter(|&&t| d.y[t] != 0.0).count();
                if pos != 1 {
                    return Err(rc_err(459, format!("variable {dep} must have exactly one positive outcome per case")));
                }
            }
        }
        Need::Rank => {}
    }
    // base alternative
    let base = if alts.is_empty() {
        0
    } else {
        match cmd.opt("basealternative", 4) {
            Some(bv) => find_alt(ds, &set, &alts, &alt_labels, bv.trim())?,
            None => {
                let mut cnt = vec![0usize; alts.len()];
                for g in &cases {
                    if let Some(&t) = g.iter().max_by(|&&a, &&b| d.y[a].total_cmp(&d.y[b])) {
                        if need == Need::One || d.y[t] != 0.0 {
                            cnt[alt_of[t]] += 1;
                        }
                    }
                }
                let mut best = 0;
                for k in 0..alts.len() {
                    if cnt[k] > cnt[best] {
                        best = k;
                    }
                }
                best
            }
        }
    };
    // model columns
    let mut names = Vec::new();
    let mut cols = Vec::new();
    let mut is_cons = Vec::new();
    for j in 0..d.k() {
        if d.omitted[j] {
            continue;
        }
        names.push(match &set.alt {
            Some(a) => format!("{a}:{}", d.names[j]),
            None => d.names[j].clone(),
        });
        cols.push(d.cols[j].clone());
        is_cons.push(false);
    }
    let n_as = cols.len();
    if set.alt.is_none() && !casevars.is_empty() {
        return Err(rc_err(198, "casevars() requires an alternatives variable"));
    }
    if set.alt.is_some() {
        let cvals: Vec<Vec<f64>> = casevars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        for k in 0..alts.len() {
            if k == base {
                continue;
            }
            let dummy: Vec<f64> = alt_of.iter().map(|&a| if a == k { 1.0 } else { 0.0 }).collect();
            for (cv, cn) in cvals.iter().zip(&casevars) {
                names.push(format!("{}:{cn}", alt_labels[k]));
                cols.push(d.rows.iter().zip(&dummy).map(|(&r, dm)| dm * cv[r]).collect());
                is_cons.push(false);
            }
            if !nocons {
                names.push(format!("{}:_cons", alt_labels[k]));
                cols.push(dummy);
                is_cons.push(true);
            }
        }
    }
    // panels
    let (case_panel, npanel) = match &set.panel {
        Some(p) => {
            let pc = util::col(ds, p)?;
            let mut keys: Vec<f64> = cases.iter().map(|g| pc[d.rows[g[0]]]).collect();
            keys.sort_by(f64::total_cmp);
            keys.dedup();
            (cases.iter().map(|g| keys.binary_search_by(|k| k.total_cmp(&pc[d.rows[g[0]]])).unwrap()).collect(), keys.len())
        }
        None => ((0..cases.len()).collect(), cases.len()),
    };
    Ok(CmData { depvar: dep, set, d, cases, case_panel, npanel, alts, alt_of, alt_labels, base, names, cols, is_cons, n_as, casevars })
}

/// Alternative given by value or label.
pub fn find_alt(ds: &DataSet, set: &CmSet, alts: &[f64], labels: &[String], spec: &str) -> Result<usize, String> {
    if let Ok(x) = spec.parse::<f64>() {
        if let Some(k) = alts.iter().position(|a| *a == x) {
            return Ok(k);
        }
    }
    if let Some(k) = labels.iter().position(|l| l == spec || l == &eq_label(spec)) {
        return Ok(k);
    }
    let _ = (ds, set);
    Err(rc_err(198, format!("alternative {spec} not found")))
}

impl CmData {
    pub fn n(&self) -> usize {
        self.d.n()
    }
    /// Linear index of row t.
    pub fn xb(&self, b: &[f64], t: usize) -> f64 {
        self.cols.iter().zip(b).map(|(c, bj)| c[t] * bj).sum()
    }
    /// Alternatives-per-case summary (min, avg, max).
    pub fn alt_sizes(&self) -> (usize, f64, usize) {
        let s: Vec<usize> = self.cases.iter().map(|g| g.len()).collect();
        (*s.iter().min().unwrap_or(&0), self.n() as f64 / self.cases.len().max(1) as f64, *s.iter().max().unwrap_or(&0))
    }
    /// Rows of a case ordered from most to least preferred, with the set
    /// of rows tied at the bottom (unranked); `higher` says whether larger
    /// outcome values are preferred.
    pub fn ranking(&self, c: usize, higher: bool) -> (Vec<usize>, Vec<usize>) {
        let mut rows = self.cases[c].clone();
        let key = |t: usize| {
            let v = self.d.y[t];
            if higher {
                -v
            } else {
                v
            }
        };
        rows.sort_by(|&a, &b| key(a).total_cmp(&key(b)));
        let worst = key(*rows.last().unwrap());
        let ranked: Vec<usize> = rows.iter().copied().filter(|&t| key(t) != worst).collect();
        let bottom: Vec<usize> = rows.iter().copied().filter(|&t| key(t) == worst).collect();
        (ranked, bottom)
    }
}

// ---------------------------------------------------------------------------
// quasi-random draws
// ---------------------------------------------------------------------------

pub const PRIMES: [u64; 20] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71];

/// Element `i` (1-based) of the Halton sequence in base `b`.
pub fn halton(mut i: u64, b: u64) -> f64 {
    let mut f = 1.0;
    let mut r = 0.0;
    while i > 0 {
        f /= b as f64;
        r += f * (i % b) as f64;
        i /= b;
    }
    r
}

/// Uniform draws: `units` x `r` x `dims`, consecutive Halton points per
/// unit after a burn-in, one prime per dimension.
pub fn halton_draws(units: usize, r: usize, dims: usize, burn: usize) -> Vec<Vec<Vec<f64>>> {
    (0..units)
        .map(|u| (0..r).map(|k| (0..dims).map(|d| halton((burn + u * r + k + 1) as u64, PRIMES[d % PRIMES.len()])).collect()).collect())
        .collect()
}

/// Number of draws: `intpoints()` or the default.
pub fn n_draws(cmd: &Command, default: usize) -> Result<usize, String> {
    Ok(util::opt_num(cmd, "intpoints", 4)?.map(|v| v.max(1.0) as usize).unwrap_or(default))
}

// ---------------------------------------------------------------------------
// GHK simulator
// ---------------------------------------------------------------------------

/// Cholesky factor of a small symmetric matrix (None if not positive
/// definite).
pub fn chol(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if !(s > 1e-12) {
                    return None;
                }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Some(l)
}

/// P(eta < m) for eta ~ N(0, L L') by the GHK simulator over the given
/// uniform draws (one vector of at least m.len() entries per draw).
pub fn ghk(m: &[f64], l: &[Vec<f64>], u: &[Vec<f64>]) -> f64 {
    let q = m.len();
    if q == 0 {
        return 1.0;
    }
    let mut tot = 0.0;
    let mut e = vec![0.0; q];
    for draw in u {
        let mut p = 1.0;
        for k in 0..q {
            let mu: f64 = (0..k).map(|j| l[k][j] * e[j]).sum();
            let ub = (m[k] - mu) / l[k][k];
            let pk = dist::normal(ub);
            p *= pk;
            if p <= 0.0 {
                break;
            }
            if k + 1 < q {
                e[k] = dist::invnormal((draw[k] * pk).clamp(1e-300, 1.0 - 1e-16));
            }
        }
        tot += p;
    }
    tot / u.len() as f64
}

// ---------------------------------------------------------------------------
// case-level likelihood with numerical derivatives
// ---------------------------------------------------------------------------

/// Log likelihood as a sum over units (cases or panels). Gradients are
/// analytic when `grad` is given, central differences otherwise; the
/// iteration Hessian is the outer product of the unit scores (or, with
/// analytic gradients, differences of the total gradient) and the final
/// one is always obtained by differencing the total gradient.
pub struct UnitLik<'a> {
    pub k: usize,
    pub units: usize,
    pub f: &'a (dyn Fn(usize, &[f64]) -> f64 + 'a),
    pub grad: Option<&'a (dyn Fn(usize, &[f64]) -> (f64, Vec<f64>) + 'a)>,
}

impl UnitLik<'_> {
    fn unit(&self, u: usize, b: &[f64]) -> (f64, Vec<f64>) {
        if let Some(g) = self.grad {
            return g(u, b);
        }
        let l0 = (self.f)(u, b);
        let mut g = vec![0.0; self.k];
        let mut x = b.to_vec();
        for j in 0..self.k {
            let s = 1e-5 * (1.0 + b[j].abs());
            x[j] = b[j] + s;
            let lp = (self.f)(u, &x);
            x[j] = b[j] - s;
            let lm = (self.f)(u, &x);
            x[j] = b[j];
            g[j] = (lp - lm) / (2.0 * s);
        }
        (l0, g)
    }
    fn total(&self, b: &[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
        let mut ll = 0.0;
        let mut g = vec![0.0; self.k];
        let mut sc = Vec::with_capacity(self.units);
        for u in 0..self.units {
            let (l, gu) = self.unit(u, b);
            ll += l;
            for j in 0..self.k {
                g[j] += gu[j];
            }
            sc.push(gu);
        }
        (ll, g, sc)
    }
    fn num_info(&self, b: &[f64]) -> Matrix {
        let k = self.k;
        let mut h = Matrix::zeros(k, k);
        let mut x = b.to_vec();
        let mut cols = Vec::with_capacity(k);
        for j in 0..k {
            let s = if self.grad.is_some() { 1e-6 } else { 1e-4 } * (1.0 + b[j].abs());
            x[j] = b[j] + s;
            let gp = self.total(&x).1;
            x[j] = b[j] - s;
            let gm = self.total(&x).1;
            x[j] = b[j];
            cols.push((0..k).map(|i| (gp[i] - gm[i]) / (2.0 * s)).collect::<Vec<f64>>());
        }
        for i in 0..k {
            for j in 0..k {
                h[(i, j)] = -(cols[j][i] + cols[i][j]) / 2.0;
            }
        }
        h
    }
}

impl Likelihood for UnitLik<'_> {
    fn k(&self) -> usize {
        self.k
    }
    fn eval(&self, b: &[f64], need_scores: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        let k = self.k;
        let (ll, g, sc) = self.total(b);
        if !ll.is_finite() {
            return (f64::NEG_INFINITY, vec![0.0; k], Matrix::identity(k), sc);
        }
        let bhhh = || {
            let mut h = Matrix::zeros(k, k);
            for s in &sc {
                for i in 0..k {
                    for j in 0..k {
                        h[(i, j)] += s[i] * s[j];
                    }
                }
            }
            h
        };
        let h = if need_scores || self.grad.is_some() {
            let h = self.num_info(b);
            if h.cholesky().is_ok() || !need_scores {
                h
            } else {
                bhhh()
            }
        } else {
            bhhh()
        };
        (ll, g, h, sc)
    }
}

/// Fit with `UnitLik` and turn the result into an `xm::Fit` with the given
/// coefficient names (aux names start with "/:").
pub fn fit_units(lik: &UnitLik, b0: Vec<f64>, names: &[String], neq: usize, naux: usize, vce: &est::Vce, cluster: Option<&[f64]>) -> Result<xm::Fit, String> {
    let res = xm::maximize(lik, b0, 300)?;
    to_fit(res, names, neq, naux, vce, cluster)
}

pub fn to_fit(res: est::MlResult, names: &[String], neq: usize, naux: usize, vce: &est::Vce, cluster: Option<&[f64]>) -> Result<xm::Fit, String> {
    let n = res.scores.len().max(1) as f64;
    let (vt, n_clust) = crate::zip::ml_vce(&res, vce, cluster, n);
    let k = names.len();
    Ok(xm::Fit {
        names: names.to_vec(),
        b: res.b.clone(),
        v: vt.clone(),
        omitted: vec![false; k],
        is_base: vec![false; k],
        bt: res.b.clone(),
        vt,
        pos: (0..k).collect(),
        n_clust,
        neq,
        naux,
        res,
    })
}

/// Number of distinct equations among coefficient names.
pub fn count_eqs(names: &[String]) -> usize {
    let mut seen: Vec<&str> = Vec::new();
    for n in names {
        if n.starts_with("/:") {
            continue;
        }
        let e = n.split_once(':').map_or("", |x| x.0);
        if !seen.contains(&e) {
            seen.push(e);
        }
    }
    seen.len().max(1)
}

/// Case-level cluster values for vce(cluster).
pub fn case_cluster(ds: &DataSet, data: &CmData, vce: &est::Vce, by_panel: bool) -> Result<Option<Vec<f64>>, String> {
    match vce {
        est::Vce::Cluster(c) => {
            let cv = util::col(ds, &crate::varlist::resolve_name(ds, c)?)?;
            if by_panel {
                let mut v = vec![0.0; data.npanel];
                for (c, g) in data.cases.iter().enumerate() {
                    v[data.case_panel[c]] = cv[data.d.rows[g[0]]];
                }
                Ok(Some(v))
            } else {
                Ok(Some(data.cases.iter().map(|g| cv[data.d.rows[g[0]]]).collect()))
            }
        }
        _ => Ok(None),
    }
}
