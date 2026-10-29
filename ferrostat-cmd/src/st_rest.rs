//! Remaining survival-analysis commands:
//!
//! * `stbase`: baseline (or as-of-time) one-record-per-subject dataset;
//! * `sttoct`: st data to count-time data (failures, censorings, entries
//!   by time and groups), then `ctset`;
//! * `sttocc`: nested case-control sampling from the risk sets;
//! * `stmc`: Mantel-Cox rate ratio (stratified on failure times);
//! * `stintcox` / `stmgintcox`: semiparametric proportional-hazards models
//!   for interval-censored data (the NPMLE of the baseline cumulative hazard
//!   on the Turnbull support, jointly with the coefficients by Newton steps
//!   on an analytic gradient); `stmgintcox` fits marginal models for several
//!   interval-censored event times with a cluster-robust variance;
//! * `estat gofplot`: Cox-Snell residual goodness-of-fit plot after stcox
//!   and streg.

use std::collections::BTreeMap;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value};
use ferrostat_parser::parser::parse_line;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, Likelihood, MlResult, Vce};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::graph::{self, Graph, Layer, LayerKind};
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::survival::{require_st, st_id, st_obs, StObs};
use crate::util;

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![
        ("stbase", Box::new(Stbase)),
        ("sttoct", Box::new(Sttoct)),
        ("sttocc", Box::new(Sttocc)),
        ("stmc", Box::new(Stmc)),
        ("stintcox", Box::new(Stintcox)),
        ("stmgintcox", Box::new(Stmgintcox)),
    ]
}

fn put_var(ds: &mut DataSet, name: &str, vals: Vec<f64>) -> Result<(), String> {
    if ds.get_var(name).is_some() {
        ds.drop_variable(name)?;
    }
    ds.add_variable(util::new_double(name, vals))
}

/// Replace the data by copies of the rows `src` (in order).
fn take_rows(ds: &mut DataSet, src: &[usize]) {
    for v in ds.vars.iter_mut() {
        v.data = src.iter().map(|&i| v.data[i].clone()).collect();
    }
    ds.n_obs = src.len();
}

/// Records grouped by subject (id or observation), each sorted by `_t`.
fn subjects(ds: &DataSet, obs: &[StObs]) -> Vec<Vec<StObs>> {
    let id = st_id(ds);
    let mut m: BTreeMap<String, Vec<StObs>> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();
    for o in obs {
        let key = match &id {
            Some(v) => util::cell_text(ds, ds.get_var(v).unwrap(), o.row, false),
            None => format!("{:012}", o.row),
        };
        if !m.contains_key(&key) {
            order.push(key.clone());
        }
        m.entry(key).or_default().push(*o);
    }
    order
        .into_iter()
        .map(|k| {
            let mut v = m.remove(&k).unwrap();
            v.sort_by(|a, b| a.t.total_cmp(&b.t));
            v
        })
        .collect()
}

// ---------------------------------------------------------------------------
// stbase
// ---------------------------------------------------------------------------

pub struct Stbase;

impl CommandHandler for Stbase {
    fn name(&self) -> &'static str {
        "stbase"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        require_st(ds)?;
        let obs = st_obs(ds, st, cmd)?;
        let subs = subjects(ds, &obs);
        let at = util::opt_num(cmd, "at", 2)?;
        let gap = cmd.opt("gap", 3).map(|s| s.trim().to_string());
        let mut src = Vec::new();
        let mut nt0 = Vec::new();
        let mut nt = Vec::new();
        let mut nd = Vec::new();
        let mut gaps = Vec::new();
        for s in &subs {
            let first = s[0];
            let last = *s.last().unwrap();
            match at {
                None => {
                    src.push(first.row);
                    nt0.push(first.t0);
                    nt.push(last.t);
                    nd.push(if last.d { 1.0 } else { 0.0 });
                    let covered: f64 = s.iter().map(|o| o.t - o.t0).sum();
                    gaps.push((last.t - first.t0) - covered);
                }
                Some(a) => {
                    if first.t0 >= a {
                        continue;
                    }
                    // the record covering time a, else the last one before it
                    let rec = s.iter().find(|o| o.t0 < a && a <= o.t).copied().unwrap_or(*s.iter().filter(|o| o.t0 < a).last().unwrap());
                    src.push(rec.row);
                    nt0.push(first.t0);
                    let fail = s.iter().find(|o| o.d && o.t <= a);
                    let exit = s.iter().map(|o| o.t).fold(f64::NEG_INFINITY, f64::max).min(a);
                    nt.push(fail.map(|o| o.t).unwrap_or(exit));
                    nd.push(if fail.is_some() { 1.0 } else { 0.0 });
                    let covered: f64 = s.iter().filter(|o| o.t0 < a).map(|o| o.t.min(a) - o.t0).sum();
                    gaps.push((exit - first.t0) - covered);
                }
            }
        }
        let nrec = ds.n_obs;
        let n = src.len();
        take_rows(ds, &src);
        put_var(ds, "_t0", nt0)?;
        put_var(ds, "_t", nt)?;
        let mut dvar = util::new_double("_d", nd);
        dvar.var_type = ferrostat_core::VarType::Byte;
        dvar.data = dvar.data.iter().map(|v| Value::Byte(v.num() as i8)).collect();
        ds.drop_variable("_d")?;
        ds.add_variable(dvar)?;
        put_var(ds, "_st", vec![1.0; n])?;
        if let Some(g) = gap {
            if ds.get_var(&g).is_some() {
                return Err(rc_err(110, format!("variable {g} already defined")));
            }
            put_var(ds, &g, gaps)?;
        }
        st.data_changed = true;
        Ok(format!("({} records, {} subjects; one record per subject{})", nrec, n, at.map(|a| format!(" as of time {}", fmt_g(a, 9))).unwrap_or_default()))
    }
}

// ---------------------------------------------------------------------------
// sttoct
// ---------------------------------------------------------------------------

pub struct Sttoct;

impl CommandHandler for Sttoct {
    fn name(&self) -> &'static str {
        "sttoct"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        require_st(ds)?;
        let names: Vec<String> = cmd.main.split_whitespace().map(String::from).collect();
        if names.len() < 2 || names.len() > 3 {
            return Err(rc_err(198, "sttoct requires failvar censvar [entvar]"));
        }
        if !cmd.has_opt("replace", 3) && st.data_changed {
            return Err(rc_err(4, "no; data in memory would be lost; specify replace"));
        }
        let by: Vec<String> = match cmd.opt("by", 2) {
            Some(b) => crate::varlist::expand(ds, b.trim())?,
            None => Vec::new(),
        };
        let obs = st_obs(ds, st, cmd)?;
        let bycols: Vec<Vec<f64>> = by.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        // (by values, time) -> [fail, cens, enter]
        let mut cells: BTreeMap<(Vec<i64>, i64), ([f64; 3], usize, f64)> = BTreeMap::new();
        let key = |r: usize| -> Vec<i64> { bycols.iter().map(|c| (c[r] * 1e6).round() as i64).collect() };
        for o in &obs {
            let k = key(o.row);
            let e = cells.entry((k.clone(), (o.t * 1e9).round() as i64)).or_insert(([0.0; 3], o.row, o.t));
            if o.d {
                e.0[0] += o.w;
            } else {
                e.0[1] += o.w;
            }
            if names.len() == 3 && o.t0 > 0.0 {
                let e = cells.entry((k, (o.t0 * 1e9).round() as i64)).or_insert(([0.0; 3], o.row, o.t0));
                e.0[2] += o.w;
            }
        }
        let src: Vec<usize> = cells.values().map(|c| c.1).collect();
        let times: Vec<f64> = cells.values().map(|c| c.2).collect();
        let counts: Vec<[f64; 3]> = cells.values().map(|c| c.0).collect();
        take_rows(ds, &src);
        let keep: Vec<String> = by.clone();
        for v in ds.varlist() {
            if !keep.contains(&v) {
                ds.drop_variable(&v)?;
            }
        }
        if let Some(c) = ds.characteristics.get_mut("_dta") {
            c.retain(|k, _| !k.starts_with("st_"));
        }
        put_var(ds, "_t", times)?;
        put_var(ds, &names[0], counts.iter().map(|c| c[0]).collect())?;
        put_var(ds, &names[1], counts.iter().map(|c| c[1]).collect())?;
        if names.len() == 3 {
            put_var(ds, &names[2], counts.iter().map(|c| c[2]).collect())?;
        }
        let mut line = format!("ctset _t {}", names.join(" "));
        if !by.is_empty() {
            line.push_str(&format!(", by({})", by.join(" ")));
        }
        let c = parse_line(&line).map_err(|e| rc_err(198, e.message))?;
        let out = crate::st_more::Ctset.execute(ds, st, &c)?;
        st.data_changed = true;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// sttocc
// ---------------------------------------------------------------------------

pub struct Sttocc;

impl CommandHandler for Sttocc {
    fn name(&self) -> &'static str {
        "sttocc"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        require_st(ds)?;
        let number = util::opt_num(cmd, "number", 1)?.unwrap_or(1.0).max(1.0) as usize;
        let matchv: Vec<String> = match cmd.opt("match", 1) {
            Some(m) => crate::varlist::expand(ds, m.trim())?,
            None => Vec::new(),
        };
        let gnames: Vec<String> = match cmd.opt("generate", 3) {
            Some(g) => g.split_whitespace().map(String::from).collect(),
            None => vec!["_set".into(), "_case".into(), "_time".into()],
        };
        if gnames.len() != 3 {
            return Err(rc_err(198, "generate() requires three names"));
        }
        let keep: Option<Vec<String>> = cmd.opt("keep", 1).map(|k| crate::varlist::expand(ds, k.trim())).transpose()?;
        let obs = st_obs(ds, st, cmd)?;
        let subs = subjects(ds, &obs);
        let mcols: Vec<Vec<f64>> = matchv.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        let mkey = |r: usize| -> Vec<i64> { mcols.iter().map(|c| (c[r] * 1e6).round() as i64).collect() };
        // failures in time order
        let mut cases: Vec<(f64, usize)> = Vec::new();
        for (si, s) in subs.iter().enumerate() {
            for o in s {
                if o.d {
                    cases.push((o.t, si));
                }
            }
        }
        cases.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut src = Vec::new();
        let mut set = Vec::new();
        let mut case = Vec::new();
        let mut time = Vec::new();
        let mut short = 0usize;
        for (k, &(t, si)) in cases.iter().enumerate() {
            let crec = subs[si].iter().find(|o| o.d && o.t == t).unwrap();
            let ck = mkey(crec.row);
            // risk set at t: other subjects with a record covering t
            let pool: Vec<usize> = subs
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != si)
                .filter_map(|(_, s)| s.iter().find(|o| o.t0 < t && t <= o.t).map(|o| o.row))
                .filter(|&r| mkey(r) == ck)
                .collect();
            let chosen: Vec<usize> = if pool.len() <= number {
                if pool.len() < number {
                    short += 1;
                }
                pool
            } else {
                // partial Fisher-Yates
                let mut p = pool;
                let mut rng = st.rng.borrow_mut();
                for i in 0..number {
                    let j = i + ((rng.uniform() * (p.len() - i) as f64) as usize).min(p.len() - i - 1);
                    p.swap(i, j);
                }
                p.truncate(number);
                p
            };
            src.push(crec.row);
            set.push((k + 1) as f64);
            case.push(1.0);
            time.push(t);
            for r in chosen {
                src.push(r);
                set.push((k + 1) as f64);
                case.push(0.0);
                time.push(t);
            }
        }
        take_rows(ds, &src);
        for v in ["_t", "_t0", "_d", "_st"] {
            let _ = ds.drop_variable(v);
        }
        if let Some(c) = ds.characteristics.get_mut("_dta") {
            c.retain(|k, _| !k.starts_with("st_"));
        }
        if let Some(keep) = keep {
            for v in ds.varlist() {
                if !keep.contains(&v) && !matchv.contains(&v) {
                    ds.drop_variable(&v)?;
                }
            }
        }
        put_var(ds, &gnames[0], set)?;
        put_var(ds, &gnames[1], case)?;
        put_var(ds, &gnames[2], time)?;
        st.data_changed = true;
        let mut out = vec![String::new(), format!("{:>20}  failure _d: case indicator {}", "", gnames[1])];
        out.push(format!("There were {} cases; {} controls were sampled per case{}.", cases.len(), number, if matchv.is_empty() { String::new() } else { format!(" (matched on {})", matchv.join(" ")) }));
        if short > 0 {
            out.push(format!("({short} risk sets had fewer than {number} available controls)"));
        }
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// stmc
// ---------------------------------------------------------------------------

pub struct Stmc;

impl CommandHandler for Stmc {
    fn name(&self) -> &'static str {
        "stmc"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        require_st(ds)?;
        let vars = crate::varlist::expand(ds, cmd.main.trim())?;
        let (var, strata) = vars.split_first().ok_or_else(|| rc_err(100, "varlist required"))?;
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        let obs = crate::survival::st_obs_mask(ds, st, &mask)?;
        let x = util::col(ds, var)?;
        let scols: Vec<Vec<f64>> = strata.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
        let mut vals: Vec<f64> = obs.iter().map(|o| x[o.row]).collect();
        vals.sort_by(f64::total_cmp);
        vals.dedup();
        let (hi, lo) = if let Some(cv) = cmd.opt("compare", 1) {
            let p: Vec<f64> = cv.split(',').map(|s| s.trim().parse::<f64>().map_err(|_| rc_err(198, "invalid compare()"))).collect::<Result<_, _>>()?;
            if p.len() != 2 {
                return Err(rc_err(198, "compare() requires two values"));
            }
            (p[0], p[1])
        } else {
            if vals.len() != 2 {
                return Err(rc_err(450, format!("{var} takes on {} values; use compare()", vals.len())));
            }
            (vals[1], vals[0])
        };
        let skey = |r: usize| -> Vec<i64> { scols.iter().map(|c| (c[r] * 1e6).round() as i64).collect() };
        let recs: Vec<(StObs, bool, Vec<i64>)> = obs.iter().filter(|o| x[o.row] == hi || x[o.row] == lo).map(|o| (*o, x[o.row] == hi, skey(o.row))).collect();
        // failure times within strata
        let mut ftimes: BTreeMap<(Vec<i64>, i64), f64> = BTreeMap::new();
        for (o, _, k) in &recs {
            if o.d {
                ftimes.insert((k.clone(), (o.t * 1e9).round() as i64), o.t);
            }
        }
        let (mut num, mut den, mut oe, mut vv) = (0.0, 0.0, 0.0, 0.0);
        for ((k, _), &t) in &ftimes {
            let (mut a, mut b, mut n1, mut n0) = (0.0, 0.0, 0.0, 0.0);
            for (o, exp, kk) in &recs {
                if kk != k || !(o.t0 < t && t <= o.t) {
                    continue;
                }
                if *exp {
                    n1 += o.w;
                    if o.d && o.t == t {
                        a += o.w;
                    }
                } else {
                    n0 += o.w;
                    if o.d && o.t == t {
                        b += o.w;
                    }
                }
            }
            let n = n1 + n0;
            if n <= 0.0 {
                continue;
            }
            let m = a + b;
            let r = a * n0 / n;
            let s = b * n1 / n;
            num += r;
            den += s;
            oe += a - m * n1 / n;
            if n > 1.0 {
                vv += m * n1 * n0 * (n - m) / (n * n * (n - 1.0));
            }
        }
        if den <= 0.0 || num <= 0.0 {
            return Err(rc_err(2000, "no failures in one of the groups"));
        }
        let rr = num / den;
        let chi = oe * oe / vv;
        let level = util::level(cmd, st)?;
        let z = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
        // test-based interval
        let s = if chi > 0.0 { rr.ln().abs() / chi.sqrt() } else { SYSMISS };
        let (lb, ub) = if is_missing(s) { (SYSMISS, SYSMISS) } else { ((rr.ln() - z * s).exp(), (rr.ln() + z * s).exp()) };
        let mut out = vec![String::new(), crate::survival::st_header(ds), String::new()];
        out.push("Mantel\u{2013}Cox estimate of the rate ratio, controlling for time".into());
        out.push(format!("    Comparing {var}=={} vs. {var}=={}", fmt_g(hi, 9), fmt_g(lo, 9)));
        if !strata.is_empty() {
            out.push(format!("    and controlling for {}", strata.join(" ")));
        }
        out.push(String::new());
        out.push(format!("{:>14} | {:>10}  {:>10}  {:>10}", "", "RR", "Lower", "Upper"));
        out.push(format!("{}+{}", "-".repeat(15), "-".repeat(38)));
        out.push(format!("{:>14} | {:>10}  {:>10}  {:>10}", "", fmt_g(rr, 8), fmt_g(lb, 8), fmt_g(ub, 8)));
        out.push(String::new());
        out.push(format!("    chi2(1) = {:>8.2}  Pr>chi2 = {:.4}", chi, dist::chi2tail(1.0, chi)));
        out.push(format!("    ({}% CI is test-based)", fmt_g(level, 4)));
        let r = st.begin_r();
        r.set("RR", rr);
        r.set("lb", lb);
        r.set("ub", ub);
        r.set("chi2", chi);
        r.set("p", dist::chi2tail(1.0, chi));
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// interval-censored Cox model
// ---------------------------------------------------------------------------

/// One interval-censored observation: failure in (l, r]; r infinite for
/// right-censored; l = 0 for left-censored. Indices into the support.
#[derive(Clone, Debug)]
struct IcObs {
    /// number of support points <= l (cumulative hazard at l)
    a: usize,
    /// number of support points <= r, None for r = infinity
    b: Option<usize>,
}

/// Turnbull support: right ends of the innermost intervals.
fn turnbull_support(ints: &[(f64, f64)]) -> Vec<f64> {
    let mut pts: Vec<(f64, u8)> = Vec::new();
    for &(l, r) in ints {
        pts.push((l, 1)); // left end (open)
        if r.is_finite() {
            pts.push((r, 0)); // right end (closed): sorts before a left end at the same value
        }
    }
    pts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut out = Vec::new();
    for w in pts.windows(2) {
        if w[0].1 == 1 && w[1].1 == 0 && w[1].0 > w[0].0 {
            out.push(w[1].0);
        }
    }
    out.dedup();
    out
}

struct IcCox<'a> {
    x: &'a [Vec<f64>],
    obs: &'a [IcObs],
    /// strata index per observation (separate baselines) and support sizes
    strata: &'a [usize],
    m: Vec<usize>,
    w: &'a [f64],
}

/// Log likelihood of one observation and its derivatives with respect to
/// (A, B, eta): values, first derivatives, second derivatives
/// [AA, AB, BB, A eta, B eta, eta eta]. B infinite for right censoring.
fn ic_obs(a: f64, b: Option<f64>, eta: f64) -> (f64, [f64; 3], [f64; 6]) {
    let u = eta.exp();
    match b {
        None => (-a * u, [-u, 0.0, -a * u], [0.0, 0.0, 0.0, -u, 0.0, -a * u]),
        Some(bb) => {
            let p = (-a * u).exp();
            let q = (-bb * u).exp();
            let r = (-(-(bb - a) * u).exp_m1()).max(1e-300); // 1 - q/p
            let d = (p * r).max(1e-300);
            let ll = -a * u + r.ln();
            let (da, db, de) = (-u * p, u * q, -a * u * p + bb * u * q);
            let (daa, dbb, dab) = (u * u * p, -u * u * q, 0.0);
            let dae = u * p * (a * u - 1.0);
            let dbe = u * q * (1.0 - bb * u);
            let dee = -a * u * p + a * a * u * u * p + bb * u * q - bb * bb * u * u * q;
            let g = [da / d, db / d, de / d];
            let h = [daa / d - g[0] * g[0], dab / d - g[0] * g[1], dbb / d - g[1] * g[1], dae / d - g[0] * g[2], dbe / d - g[1] * g[2], dee / d - g[2] * g[2]];
            (ll, g, h)
        }
    }
}

impl IcCox<'_> {
    fn kx(&self) -> usize {
        self.x.len()
    }
    fn offset(&self, s: usize) -> usize {
        self.kx() + self.m[..s].iter().sum::<usize>()
    }
    fn lambdas(&self, b: &[f64]) -> Vec<Vec<f64>> {
        (0..self.m.len()).map(|s| (0..self.m[s]).map(|k| b[self.offset(s) + k].exp()).collect()).collect()
    }
    /// (ll, gradient, negative Hessian, scores)
    fn derivs(&self, b: &[f64], need_scores: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        let kx = self.kx();
        let np = b.len();
        let lam = self.lambdas(b);
        let cum: Vec<Vec<f64>> = lam
            .iter()
            .map(|l| {
                let mut acc = 0.0;
                let mut v = vec![0.0];
                for x in l {
                    acc += x;
                    v.push(acc);
                }
                v
            })
            .collect();
        let ns = self.m.len();
        let mut ll = 0.0;
        let mut g = vec![0.0; np];
        let mut h = Matrix::zeros(np, np);
        // per stratum endpoint accumulators
        let mut ge: Vec<Vec<f64>> = self.m.iter().map(|&m| vec![0.0; m + 1]).collect();
        let mut ce: Vec<Vec<f64>> = self.m.iter().map(|&m| vec![0.0; (m + 1) * (m + 1)]).collect();
        let mut xe: Vec<Vec<Vec<f64>>> = self.m.iter().map(|&m| vec![vec![0.0; kx]; m + 1]).collect();
        let mut scores = Vec::new();
        for i in 0..self.obs.len() {
            let o = &self.obs[i];
            let s = self.strata[i];
            let w = self.w[i];
            let eta: f64 = (0..kx).map(|j| self.x[j][i] * b[j]).sum();
            let av = cum[s][o.a];
            let bv = o.b.map(|q| cum[s][q]);
            let (l, d1, d2) = ic_obs(av, bv, eta);
            ll += w * l;
            for j in 0..kx {
                g[j] += w * d1[2] * self.x[j][i];
                for k in 0..=j {
                    h[(j, k)] -= w * d2[5] * self.x[j][i] * self.x[k][i];
                }
            }
            let m1 = self.m[s] + 1;
            ge[s][o.a] += w * d1[0];
            ce[s][o.a * m1 + o.a] += w * d2[0];
            for j in 0..kx {
                xe[s][o.a][j] += w * d2[3] * self.x[j][i];
            }
            if let Some(q) = o.b {
                ge[s][q] += w * d1[1];
                ce[s][o.a * m1 + q] += w * d2[1];
                ce[s][q * m1 + o.a] += w * d2[1];
                ce[s][q * m1 + q] += w * d2[2];
                for j in 0..kx {
                    xe[s][q][j] += w * d2[4] * self.x[j][i];
                }
            }
            if need_scores {
                let mut sc = vec![0.0; np];
                for j in 0..kx {
                    sc[j] = d1[2] * self.x[j][i];
                }
                let off = self.offset(s);
                for k in 0..self.m[s] {
                    let mut v = 0.0;
                    if k < o.a {
                        v += d1[0];
                    }
                    if o.b.is_some_and(|q| k < q) {
                        v += d1[1];
                    }
                    sc[off + k] = v * lam[s][k];
                }
                scores.push(sc);
            }
        }
        for s in 0..ns {
            let m = self.m[s];
            let m1 = m + 1;
            let off = self.offset(s);
            // suffix sums over endpoints p > k
            let mut gs = vec![0.0; m1 + 1];
            let mut xs = vec![vec![0.0; kx]; m1 + 1];
            for p in (0..m1).rev() {
                gs[p] = gs[p + 1] + ge[s][p];
                for j in 0..kx {
                    xs[p][j] = xs[p + 1][j] + xe[s][p][j];
                }
            }
            // 2-D suffix sums of ce
            let mut ss = vec![0.0; (m1 + 1) * (m1 + 1)];
            for p in (0..m1).rev() {
                for q in (0..m1).rev() {
                    ss[p * (m1 + 1) + q] = ce[s][p * m1 + q] + ss[(p + 1) * (m1 + 1) + q] + ss[p * (m1 + 1) + q + 1] - ss[(p + 1) * (m1 + 1) + q + 1];
                }
            }
            let l = &lam[s];
            for k in 0..m {
                let gk = gs[k + 1] * l[k];
                g[off + k] += gk;
                for j in 0..kx {
                    h[(off + k, j)] -= xs[k + 1][j] * l[k];
                }
                for q in 0..=k {
                    h[(off + k, off + q)] -= l[k] * l[q] * ss[(k + 1) * (m1 + 1) + q + 1];
                }
                h[(off + k, off + k)] -= gk;
            }
        }
        for i in 0..np {
            for j in 0..i {
                h[(j, i)] = h[(i, j)];
            }
        }
        (ll, g, h, scores)
    }
}

impl Likelihood for IcCox<'_> {
    fn k(&self) -> usize {
        self.kx() + self.m.iter().sum::<usize>()
    }
    fn eval(&self, b: &[f64], need_scores: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        self.derivs(b, need_scores)
    }
}

/// Fit: coefficients (first kx entries), info, scores.
fn ic_fit(x: &[Vec<f64>], ints_by_stratum: &[Vec<(usize, f64, f64)>], n: usize, w: &[f64]) -> Result<(MlResult, Vec<usize>, Vec<Vec<f64>>), String> {
    let mut obs = vec![IcObs { a: 0, b: None }; n];
    let mut strata = vec![0usize; n];
    let mut m = Vec::new();
    let mut supports = Vec::new();
    for (s, ints) in ints_by_stratum.iter().enumerate() {
        let sup = turnbull_support(&ints.iter().map(|x| (x.1, x.2)).collect::<Vec<_>>());
        if sup.is_empty() {
            return Err(rc_err(2000, "no failures (no finite interval endpoints)"));
        }
        for &(i, l, r) in ints {
            let a = sup.iter().filter(|&&p| p <= l).count();
            let b = if r.is_finite() { Some(sup.iter().filter(|&&p| p <= r).count()) } else { None };
            if b == Some(a) {
                return Err(rc_err(498, "an interval contains no support point"));
            }
            obs[i] = IcObs { a, b };
            strata[i] = s;
        }
        m.push(sup.len());
        supports.push(sup);
    }
    let model = IcCox { x, obs: &obs, strata: &strata, m: m.clone(), w };
    let kx = x.len();
    let mut b0 = vec![0.0; kx];
    for &ms in &m {
        b0.extend(std::iter::repeat((1.0 / ms as f64).ln()).take(ms));
    }
    let res = crate::xmodels::maximize(&model, b0, 400)?;
    Ok((res, m, supports))
}

/// `(l, r)` analysis intervals from two variables (missing upper = right-censored,
/// missing or zero lower = left-censored, equal = exact: (t - eps, t]).
fn intervals(lv: &[f64], uv: &[f64], rows: &[usize]) -> Result<Vec<(f64, f64)>, String> {
    rows.iter()
        .map(|&r| {
            let l = if is_missing(lv[r]) { 0.0 } else { lv[r].max(0.0) };
            let u = if is_missing(uv[r]) { f64::INFINITY } else { uv[r] };
            if u < l {
                return Err(rc_err(498, "interval upper endpoint less than lower endpoint"));
            }
            if u == l {
                Ok(((l - 1e-9 * l.abs().max(1.0)).max(0.0), u))
            } else {
                Ok((l, u))
            }
        })
        .collect()
}

fn ic_header(title: &str, n: usize, counts: [usize; 4], chi2: f64, df: f64, ll: f64, robust: bool, clust: Option<(usize, String)>) -> Vec<String> {
    let mut out = vec![String::new(), crate::choice_mixed::hrow(title, "Number of obs", util::commas(n))];
    out.push(format!("{:<48}{:>16} = {:>10}", "", "Uncensored", util::commas(counts[0])));
    out.push(format!("{:<48}{:>16} = {:>10}", "", "Left-censored", util::commas(counts[1])));
    out.push(format!("{:<48}{:>16} = {:>10}", "", "Right-censored", util::commas(counts[2])));
    out.push(format!("{:<48}{:>16} = {:>10}", "", "Interval-cens.", util::commas(counts[3])));
    out.push(String::new());
    out.push(crate::choice_mixed::hrow("", &format!("Wald chi2({})", fmt_g(df, 9)), crate::xmodels::f2(chi2)));
    out.push(crate::choice_mixed::hrow(&format!("Log {}likelihood = {}", if robust { "pseudo" } else { "" }, fmt_g(ll, 9)), "Prob > chi2", crate::xmodels::f4(crate::xmodels::chi2p(df, chi2))));
    if let Some((g, v)) = clust {
        out.push(format!("{:>78}", format!("(Std. err. adjusted for {} clusters in {v})", util::commas(g))));
    }
    out.push(String::new());
    out
}

fn censor_counts(ints: &[(f64, f64)], lv: &[f64], uv: &[f64], rows: &[usize]) -> [usize; 4] {
    let mut c = [0usize; 4];
    for (k, &r) in rows.iter().enumerate() {
        let (l, u) = (lv[r], uv[r]);
        let _ = ints[k];
        if !is_missing(l) && !is_missing(u) && l == u {
            c[0] += 1;
        } else if is_missing(l) || l <= 0.0 {
            c[1] += 1;
        } else if is_missing(u) {
            c[2] += 1;
        } else {
            c[3] += 1;
        }
    }
    c
}

pub struct Stintcox;

impl CommandHandler for Stintcox {
    fn name(&self) -> &'static str {
        "stintcox"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if cmd.main.trim().is_empty() && cmd.opts.is_empty() {
            return crate::models::replay(st, "stintcox");
        }
        let iv = cmd.opt("interval", 3).ok_or_else(|| rc_err(198, "option interval() required"))?;
        let ivars: Vec<String> = crate::varlist::expand(ds, iv.trim())?;
        if ivars.len() != 2 {
            return Err(rc_err(198, "interval() requires two variables"));
        }
        // sample: covariates nonmissing; endpoints may be missing (censoring)
        let mut d = est::build(ds, st, cmd, est::DesignSpec { depvar: None, indep: cmd.main.trim(), noconstant: true, weights: "fpi", extra: vec![] })?;
        let lv = util::col(ds, &ivars[0])?;
        let uv = util::col(ds, &ivars[1])?;
        let keep: Vec<bool> = d.rows.iter().map(|&r| !(is_missing(lv[r]) && is_missing(uv[r]))).collect();
        d = crate::xmodels::subset(&d, &keep);
        let rows = d.rows.clone();
        let n = rows.len();
        let ints = intervals(&lv, &uv, &rows)?;
        let vce = est::parse_vce(cmd, Vce::Oim)?;
        let cluster = est::cluster_values(ds, &d, &vce)?;
        let act = d.active();
        let x: Vec<Vec<f64>> = act.iter().map(|&j| d.cols[j].clone()).collect();
        let strat: Vec<Vec<(usize, f64, f64)>> = vec![ints.iter().enumerate().map(|(i, &(l, r))| (i, l, r)).collect()];
        let (res, m, _) = ic_fit(&x, &strat, n, &d.w)?;
        let kx = act.len();
        let (vfull, nclust) = crate::zip::ml_vce(&res, &vce, cluster.as_deref(), n as f64);
        let mut vt = Matrix::zeros(kx, kx);
        for i in 0..kx {
            for j in 0..kx {
                vt[(i, j)] = vfull[(i, j)];
            }
        }
        let k = d.k();
        let mut b = vec![0.0; k];
        for (a, &j) in act.iter().enumerate() {
            b[j] = res.b[a];
        }
        let v = est::expand_v(&vt, &act, k);
        let (chi2, df) = est::wald_all(&d, &b, &v);
        let level = util::level(cmd, st)?;
        let nohr = cmd.has_opt("nohr", 4);
        let counts = censor_counts(&ints, &lv, &uv, &rows);
        let robust = vce != Vce::Oim;
        let clab = match (&vce, nclust) {
            (Vce::Cluster(c), Some(g)) => Some((g, c.clone())),
            _ => None,
        };
        let mut out = est::iteration_log(&res.iterations, "Log likelihood");
        out.extend(ic_header("Cox regression with interval-censored data", n, counts, chi2, df, res.ll, robust, clab));
        out.extend(est::coef_table(&est::TableSpec { depvar: "", names: &d.names, b: &b, v: &v, omitted: &d.omitted, is_base: &d.is_base, df: None, level, eform: if nohr { None } else { Some("Haz. ratio") }, coef_title: "Coefficient" }));
        if !res.converged {
            out.push("convergence not achieved".into());
        }
        let text = out.join("\n");
        let ll = res.ll;
        let conv = res.converged;
        let ivtxt = ivars.join(" ");
        est::post(st, "stintcox", &d, &b, &v, &|e: &mut Results| {
            e.set("ll", ll);
            e.set("chi2", chi2);
            e.set("df_m", df);
            e.set("p", crate::xmodels::chi2p(df, chi2));
            e.set("N_uncens", counts[0] as f64);
            e.set("N_lc", counts[1] as f64);
            e.set("N_rc", counts[2] as f64);
            e.set("N_int", counts[3] as f64);
            e.set("N_supp", m[0] as f64);
            e.set("converged", if conv { 1.0 } else { 0.0 });
            e.set_str("depvar", ivtxt.clone());
            e.set_str("title", "Cox regression with interval-censored data");
            e.set_str("vce", est::vce_label(&vce).0);
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

/// `stmgintcox (l1 u1 = xvars) (l2 u2 = xvars) ..., id(idvar)`: marginal
/// proportional-hazards models for several interval-censored events, each
/// with its own baseline and coefficients, and a variance clustered on the
/// subject (independence working likelihood).
pub struct Stmgintcox;

impl CommandHandler for Stmgintcox {
    fn name(&self) -> &'static str {
        "stmgintcox"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let groups = crate::xmodels::paren_groups(cmd.main.trim());
        if groups.is_empty() {
            return crate::models::replay(st, "stmgintcox");
        }
        let mut specs = Vec::new();
        let mut extra: Vec<String> = Vec::new();
        for g in &groups {
            let (lhs, rhs) = g.split_once('=').ok_or_else(|| rc_err(198, "equations must be (lower upper = indepvars)"))?;
            let ends: Vec<String> = crate::varlist::expand(ds, lhs.trim())?;
            if ends.len() != 2 {
                return Err(rc_err(198, "each equation needs a lower and an upper endpoint variable"));
            }
            extra.extend(crate::zip::base_vars(ds, rhs));
            specs.push((ends, rhs.trim().to_string()));
        }
        let idv = cmd.opt("id", 2).map(|s| crate::varlist::resolve_name(ds, s.trim())).transpose()?;
        if let Some(i) = &idv {
            extra.push(i.clone());
        }
        // common sample over all covariates
        let designs: Vec<est::Design> = specs.iter().map(|(_, rhs)| est::build(ds, st, cmd, est::DesignSpec { depvar: None, indep: rhs, noconstant: true, weights: "", extra: extra.clone() })).collect::<Result<_, _>>()?;
        let rows = designs[0].rows.clone();
        let n0 = rows.len();
        // stack: one record per (subject, event) with an observed interval
        let mut names = Vec::new();
        let mut xcols: Vec<Vec<f64>> = Vec::new();
        let mut strat: Vec<Vec<(usize, f64, f64)>> = Vec::new();
        let mut clus: Vec<f64> = Vec::new();
        let mut counts = [0usize; 4];
        let idcol = match &idv {
            Some(i) => util::col(ds, i)?,
            None => (0..ds.n_obs).map(|r| r as f64).collect(),
        };
        let mut nrec = 0usize;
        let mut rec_rows: Vec<Vec<(usize, usize)>> = Vec::new(); // per event: (stacked index, design row)
        for (e, ((ends, _), d)) in specs.iter().zip(&designs).enumerate() {
            let lv = util::col(ds, &ends[0])?;
            let uv = util::col(ds, &ends[1])?;
            let mut recs = Vec::new();
            for (t, &r) in rows.iter().enumerate() {
                if is_missing(lv[r]) && is_missing(uv[r]) {
                    continue;
                }
                recs.push((nrec, t));
                nrec += 1;
            }
            let rr: Vec<usize> = recs.iter().map(|x| rows[x.1]).collect();
            let ints = intervals(&lv, &uv, &rr)?;
            let c = censor_counts(&ints, &lv, &uv, &rr);
            for q in 0..4 {
                counts[q] += c[q];
            }
            strat.push(recs.iter().zip(&ints).map(|(x, iv)| (x.0, iv.0, iv.1)).collect());
            for &j in &d.active() {
                names.push(format!("{}:{}", ends[1], d.names[j]));
            }
            let _ = e;
            rec_rows.push(recs);
        }
        // block-diagonal covariates over stacked records
        for (e, d) in designs.iter().enumerate() {
            for &j in &d.active() {
                let mut col = vec![0.0; nrec];
                for &(s, t) in &rec_rows[e] {
                    col[s] = d.cols[j][t];
                }
                xcols.push(col);
            }
        }
        for recs in &rec_rows {
            for &(_, t) in recs {
                clus.push(idcol[rows[t]]);
            }
        }
        let w = vec![1.0; nrec];
        let (res, _, _) = ic_fit(&xcols, &strat, nrec, &w)?;
        let kx = xcols.len();
        let (hinv, _) = res.info.sym_inverse(1e-14);
        let (vfull, g) = est::sandwich(&hinv, &res.scores, Some(&clus), 1.0);
        let gf = g as f64;
        let vfull = vfull.scale(gf / (gf - 1.0));
        let mut v = Matrix::zeros(kx, kx);
        for i in 0..kx {
            for j in 0..kx {
                v[(i, j)] = vfull[(i, j)];
            }
        }
        let b: Vec<f64> = res.b[..kx].to_vec();
        let mut dpost = designs[0].clone();
        dpost.names = names.clone();
        dpost.omitted = vec![false; kx];
        dpost.is_base = vec![false; kx];
        dpost.cols = xcols.iter().map(|_| Vec::new()).collect();
        let (chi2, df) = crate::heckman::wald(&b, &v, &(0..kx).collect::<Vec<_>>());
        let level = util::level(cmd, st)?;
        let nohr = cmd.has_opt("nohr", 4);
        let mut out = est::iteration_log(&res.iterations, "Log pseudolikelihood");
        out.extend(ic_header("Marginal Cox regression with interval-censored data", nrec, counts, chi2, df, res.ll, true, Some((g, idv.clone().unwrap_or_else(|| "_n".into())))));
        out.extend(est::coef_table(&est::TableSpec { depvar: "", names: &names, b: &b, v: &v, omitted: &vec![false; kx], is_base: &vec![false; kx], df: None, level, eform: if nohr { None } else { Some("Haz. ratio") }, coef_title: "Coefficient" }));
        let text = out.join("\n");
        let ll = res.ll;
        let neq = specs.len();
        est::post(st, "stmgintcox", &dpost, &b, &v, &|e: &mut Results| {
            e.set("N", nrec as f64);
            e.set("N_sub", n0 as f64);
            e.set("N_clust", gf);
            e.set("k_eq", neq as f64);
            e.set("ll", ll);
            e.set("chi2", chi2);
            e.set("df_m", df);
            e.set("p", crate::xmodels::chi2p(df, chi2));
            e.set_str("vce", "cluster");
            e.set_str("vcetype", "Robust");
            e.set_str("title", "Marginal Cox regression with interval-censored data");
            e.set_str("_output", text.clone());
        });
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// estat gofplot
// ---------------------------------------------------------------------------

fn gofplot(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let tmp = "__gof_cs";
    let pc = parse_line(&format!("predict double {tmp}, csnell")).map_err(|e| rc_err(198, e.message))?;
    crate::postest::Predict.execute(ds, st, &pc)?;
    let cs = util::col(ds, tmp)?;
    let dcol = util::col(ds, "_d");
    let _ = ds.drop_variable(tmp);
    let dcol = dcol?;
    let sample = st.e.sample.clone().unwrap_or_else(|| vec![true; ds.n_obs]);
    let mut pts: Vec<(f64, bool)> = (0..ds.n_obs).filter(|&i| sample.get(i).copied().unwrap_or(false) && !is_missing(cs[i])).map(|i| (cs[i], dcol[i] == 1.0)).collect();
    if pts.is_empty() {
        return Err(rc_err(2000, "no observations"));
    }
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Nelson-Aalen cumulative hazard of the residuals
    let n = pts.len();
    let mut h = 0.0;
    let mut line = vec![(0.0, 0.0)];
    let mut i = 0;
    while i < n {
        let r = pts[i].0;
        let mut j = i;
        let mut d = 0.0;
        while j < n && pts[j].0 == r {
            if pts[j].1 {
                d += 1.0;
            }
            j += 1;
        }
        let at_risk = (n - i) as f64;
        if d > 0.0 {
            line.push((r, h));
            h += d / at_risk;
            line.push((r, h));
        }
        i = j;
    }
    let maxr = line.last().map(|p| p.0).unwrap_or(1.0);
    let ref_line = Layer { kind: LayerKind::Line, points: vec![(0.0, 0.0), (maxr, maxr)], label: "Reference".into(), pattern: "dash".into(), ..Default::default() };
    let na = Layer { kind: LayerKind::Line, points: line, label: "Cumulative hazard".into(), ..Default::default() };
    let mut g = Graph { layers: vec![na, ref_line], xtitle: "Cox\u{2013}Snell residual".into(), ytitle: "Cumulative hazard".into(), title: "Cox\u{2013}Snell residuals goodness-of-fit plot".into(), legend: true, ..Default::default() };
    st.r = Results::new();
    st.r.set("H_max", h);
    st.r.set("r_max", maxr);
    graph::finish(&mut g, cmd, st)
}

/// `estat` subcommands provided here (None if not ours).
pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    match sub {
        "gofplot" | "gofp" if ecmd == "cox" || ecmd == "streg" || ecmd.starts_with("ereg") || ecmd.starts_with("weib") || st.e.macros.get("cmd2").map(String::as_str) == Some("streg") => Some(gofplot(ds, st, cmd)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turnbull_support_of_overlapping_intervals() {
        // (0,2], (1,3], (4,inf): innermost (1,2] -> support {2}
        let s = turnbull_support(&[(0.0, 2.0), (1.0, 3.0), (4.0, f64::INFINITY)]);
        assert_eq!(s, vec![2.0]);
        // exact-like intervals give one point each
        let s = turnbull_support(&[(0.9, 1.0), (1.9, 2.0)]);
        assert_eq!(s, vec![1.0, 2.0]);
    }

    #[test]
    fn ic_gradient_and_hessian_match_numeric() {
        let x = vec![vec![0.5, -1.0, 2.0, 0.0, 1.0]];
        let obs = vec![IcObs { a: 0, b: Some(1) }, IcObs { a: 1, b: Some(2) }, IcObs { a: 2, b: None }, IcObs { a: 0, b: Some(2) }, IcObs { a: 1, b: None }];
        let strata = vec![0; 5];
        let w = vec![1.0; 5];
        let m = IcCox { x: &x, obs: &obs, strata: &strata, m: vec![2], w: &w };
        let b = vec![0.3, -0.7, -0.2];
        let (_, g, h, _) = m.derivs(&b, false);
        for j in 0..3 {
            let mut bp = b.clone();
            let mut bm = b.clone();
            bp[j] += 1e-6;
            bm[j] -= 1e-6;
            let num = (m.derivs(&bp, false).0 - m.derivs(&bm, false).0) / 2e-6;
            assert!((num - g[j]).abs() < 1e-5, "{j}: {num} vs {}", g[j]);
            let (gp, gm) = (m.derivs(&bp, false).1, m.derivs(&bm, false).1);
            for i in 0..3 {
                let nh = -(gp[i] - gm[i]) / 2e-6;
                assert!((nh - h[(i, j)]).abs() < 1e-4, "h {i}{j}: {nh} vs {}", h[(i, j)]);
            }
        }
    }
}
