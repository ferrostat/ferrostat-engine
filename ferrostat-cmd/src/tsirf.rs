//! Impulse-response functions, dynamic multipliers and forecast-error
//! variance decompositions after `var`, `svar` and `vec`: `irf create`,
//! `irf set`, `irf table`, `irf ctable`, `irf graph`, `irf ograph`,
//! `irf cgraph`, `irf describe`, `irf drop`, `irf rename` (also as `varirf`).
//!
//! Asymptotic standard errors use the delta method: the statistics are
//! smooth functions of the lag coefficients, vech(Σ) and, after `svar`, of
//! vec(A, B), whose joint VCE is block diagonal (e(V) of the VAR,
//! 2 D⁺(Σ⊗Σ)D⁺'/T and e(V) of svar); the Jacobian is computed numerically.
//! `bs`/`bsp` give bootstrap standard errors after `var`.
//!
//! IRF files are plain text, written through the virtual file system:
//!
//! ```text
//! ferrostat-irf 1
//! result <irfname>
//! meta <key> <value>          (model, order, step, stderror, lags, ...)
//! endog <names>
//! exog <names>                (impulses of dm/cdm)
//! stat <name> <values>        (irf oirf dm cirf coirf cdm fevd sirf csirf
//!                              sfevd and std<name>; values ordered by
//!                              impulse, response, step; "." = missing)
//! end
//! ```
//!
//! Fields are separated by tabs.

use std::collections::BTreeMap;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::vfs;
use ferrostat_core::DataSet;
use ferrostat_parser::parser::{parse_line, split_top_level_words};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::graph::{Graph, Layer, LayerKind, RangeStyle};
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::tsla::{chol, inv, mul, vech};
use crate::tsvar;
use crate::util;

pub const STATS: [&str; 10] = ["irf", "oirf", "dm", "cirf", "coirf", "cdm", "fevd", "sirf", "csirf", "sfevd"];

fn stat_label(s: &str) -> &'static str {
    match s {
        "irf" => "impulse-response function (irf)",
        "oirf" => "orthogonalized irf",
        "dm" => "dynamic-multiplier function",
        "cirf" => "cumulative irf",
        "coirf" => "cumulative orthogonalized irf",
        "cdm" => "cumulative dynamic multiplier",
        "fevd" => "fraction of mse due to impulse",
        "sirf" => "structural irf",
        "csirf" => "cumulative structural irf",
        "sfevd" => "(structural) fraction of mse due to impulse",
        _ => "",
    }
}

// ---------------------------------------------------------------------------
// IRF results and files
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct IrfResult {
    pub name: String,
    pub meta: Vec<(String, String)>,
    pub endog: Vec<String>,
    pub exog: Vec<String>,
    pub step: usize,
    /// statistic (and std<statistic>) -> values [impulse][response][step]
    pub vals: BTreeMap<String, Vec<f64>>,
}

impl IrfResult {
    pub fn meta(&self, key: &str) -> String {
        self.meta.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone()).unwrap_or_default()
    }
    fn impulses(&self, stat: &str) -> &[String] {
        if stat.ends_with("dm") { &self.exog } else { &self.endog }
    }
    /// Value of `stat` for impulse `i`, response `j`, step `h`.
    pub fn get(&self, stat: &str, i: usize, j: usize, h: usize) -> f64 {
        let base = stat.trim_start_matches("std");
        let _ = base;
        let k = self.endog.len();
        match self.vals.get(stat) {
            Some(v) => v.get((i * k + j) * (self.step + 1) + h).copied().unwrap_or(SYSMISS),
            None => SYSMISS,
        }
    }
    pub fn has(&self, stat: &str) -> bool {
        self.vals.contains_key(stat)
    }
}

#[derive(Clone, Debug, Default)]
pub struct IrfFile {
    pub results: Vec<IrfResult>,
    pub created: String,
}

thread_local! {
    /// Text of the IRF files written by this thread, so that a session keeps
    /// reading its own results even if another session rewrites the file.
    static IRF_CACHE: std::cell::RefCell<std::collections::HashMap<String, String>> = std::cell::RefCell::new(Default::default());
}

fn num_text(v: f64) -> String {
    if is_missing(v) { ".".into() } else { format!("{v:e}") }
}

impl IrfFile {
    pub fn to_text(&self) -> String {
        let mut s = String::from("ferrostat-irf\t1\n");
        if !self.created.is_empty() {
            s.push_str(&format!("created\t{}\n", self.created));
        }
        for r in &self.results {
            s.push_str(&format!("result\t{}\n", r.name));
            for (k, v) in &r.meta {
                s.push_str(&format!("meta\t{k}\t{v}\n"));
            }
            s.push_str(&format!("endog\t{}\n", r.endog.join(" ")));
            s.push_str(&format!("exog\t{}\n", r.exog.join(" ")));
            s.push_str(&format!("step\t{}\n", r.step));
            for (k, v) in &r.vals {
                s.push_str(&format!("stat\t{k}\t{}\n", v.iter().map(|x| num_text(*x)).collect::<Vec<_>>().join(" ")));
            }
            s.push_str("end\n");
        }
        s
    }
    pub fn parse(text: &str) -> Result<IrfFile, String> {
        let mut lines = text.lines();
        if !lines.next().is_some_and(|l| l.starts_with("ferrostat-irf")) {
            return Err(rc_err(610, "file is not an IRF file"));
        }
        let mut f = IrfFile::default();
        let mut cur: Option<IrfResult> = None;
        for line in lines {
            let mut parts = line.splitn(3, '\t');
            let tag = parts.next().unwrap_or("");
            let a = parts.next().unwrap_or("");
            let b = parts.next().unwrap_or("");
            match tag {
                "created" => f.created = a.to_string(),
                "result" => cur = Some(IrfResult { name: a.to_string(), ..Default::default() }),
                "meta" => cur.iter_mut().for_each(|c| c.meta.push((a.to_string(), b.to_string()))),
                "endog" => cur.iter_mut().for_each(|c| c.endog = a.split_whitespace().map(String::from).collect()),
                "exog" => cur.iter_mut().for_each(|c| c.exog = a.split_whitespace().map(String::from).collect()),
                "step" => cur.iter_mut().for_each(|c| c.step = a.trim().parse().unwrap_or(0)),
                "stat" => {
                    let vals: Vec<f64> = b.split_whitespace().map(|x| if x == "." { SYSMISS } else { x.parse().unwrap_or(SYSMISS) }).collect();
                    cur.iter_mut().for_each(|c| {
                        c.vals.insert(a.to_string(), vals.clone());
                    });
                }
                "end" => {
                    if let Some(c) = cur.take() {
                        f.results.push(c);
                    }
                }
                _ => {}
            }
        }
        Ok(f)
    }
    pub fn load(path: &str) -> Result<IrfFile, String> {
        let cached = IRF_CACHE.with(|c| c.borrow().get(path).cloned());
        let text = match cached {
            Some(t) if vfs::exists(path) => t,
            _ => vfs::read_to_string(path).map_err(|_| rc_err(601, format!("file {path} not found")))?,
        };
        IrfFile::parse(&text)
    }
    pub fn save(&self, path: &str) -> Result<(), String> {
        let text = self.to_text();
        vfs::write(path, text.as_bytes()).map_err(|e| rc_err(603, format!("file {path} could not be opened: {e}")))?;
        IRF_CACHE.with(|c| c.borrow_mut().insert(path.to_string(), text));
        Ok(())
    }
    pub fn get(&self, name: &str) -> Option<&IrfResult> {
        self.results.iter().find(|r| r.name == name)
    }
}

fn irf_path(name: &str) -> String {
    let n = util::unquote(name.trim());
    let base = n.rsplit(['/', '\\']).next().unwrap_or(&n);
    if base.contains('.') { n } else { format!("{n}.irf") }
}

fn active_file(st: &Store) -> Option<String> {
    st.globals.get("S_vrffile").filter(|s| !s.is_empty()).cloned()
}

fn now() -> String {
    chrono::Local::now().format("%d %b %Y %H:%M").to_string()
}

/// `set(file[, replace])`: make the file active, creating it if needed.
fn set_file(st: &mut Store, spec: &str, msgs: &mut Vec<String>) -> Result<String, String> {
    let (file, opts) = match spec.split_once(',') {
        Some((f, o)) => (f.trim(), o.trim()),
        None => (spec.trim(), ""),
    };
    let path = irf_path(file);
    let replace = opts.split_whitespace().any(|o| "replace".starts_with(o) && o.len() >= 3);
    if replace || !vfs::exists(&path) {
        IrfFile { results: Vec::new(), created: now() }.save(&path)?;
        msgs.push(format!("(file {path} created)"));
    }
    st.globals.insert("S_vrffile".into(), path.clone());
    msgs.push(format!("(file {path} now active)"));
    Ok(path)
}

fn file_for(st: &mut Store, cmd: &Command, msgs: &mut Vec<String>) -> Result<String, String> {
    match cmd.opt("set", 3) {
        Some(s) => set_file(st, s, msgs),
        None => active_file(st).ok_or_else(|| rc_err(198, "no irf file active")),
    }
}

// ---------------------------------------------------------------------------
// Computation
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Var,
    Svar,
    Vec,
}

/// Parameters of the moving-average representation.
struct Model {
    kind: Kind,
    k: usize,
    p: usize,
    /// exogenous impulses (dynamic multipliers): bases and max lag
    exog: Vec<String>,
    s: usize,
    theta: Vec<f64>,
    vtheta: Option<Matrix>,
    endog: Vec<String>,
    t: f64,
    lags: String,
}

impl Model {
    fn off_b(&self) -> usize {
        self.k * self.k * self.p
    }
    fn off_sig(&self) -> usize {
        self.off_b() + self.k * self.exog.len() * (self.s + 1)
    }
    fn off_ab(&self) -> usize {
        self.off_sig() + self.k * (self.k + 1) / 2
    }
    fn n_theta(&self) -> usize {
        self.off_ab() + if self.kind == Kind::Svar { 2 * self.k * self.k } else { 0 }
    }
    fn avail(&self) -> Vec<&'static str> {
        let mut v = vec!["irf", "oirf"];
        if !self.exog.is_empty() {
            v.push("dm");
        }
        v.extend(["cirf", "coirf"]);
        if !self.exog.is_empty() {
            v.push("cdm");
        }
        v.push("fevd");
        if self.kind == Kind::Svar {
            v.extend(["sirf", "csirf", "sfevd"]);
        }
        v
    }

    /// All statistics for steps 0..=h, concatenated in `avail()` order.
    fn compute(&self, th: &[f64], h: usize, order: &[usize]) -> Vec<f64> {
        let k = self.k;
        let p = self.p;
        let a: Vec<Matrix> = (0..p)
            .map(|l| {
                let mut m = Matrix::zeros(k, k);
                for i in 0..k {
                    for j in 0..k {
                        m[(i, j)] = th[l * k * k + i * k + j];
                    }
                }
                m
            })
            .collect();
        let r = self.exog.len();
        let bx: Vec<Matrix> = (0..=self.s)
            .map(|l| {
                let mut m = Matrix::zeros(k, r);
                for i in 0..k {
                    for j in 0..r {
                        m[(i, j)] = th[self.off_b() + l * k * r + i * r + j];
                    }
                }
                m
            })
            .collect();
        let sigma = crate::tsla::unvech(&th[self.off_sig()..self.off_sig() + k * (k + 1) / 2], k);
        // Φ_i
        let mut phi = vec![Matrix::identity(k)];
        for i in 1..=h {
            let mut m = Matrix::zeros(k, k);
            for j in 1..=i.min(p) {
                m = crate::tsla::add(&m, &mul(&phi[i - j], &a[j - 1]));
            }
            phi.push(m);
        }
        // Cholesky factor in the requested order
        let mut sp = Matrix::zeros(k, k);
        for x in 0..k {
            for y in 0..k {
                sp[(x, y)] = sigma[(order[x], order[y])];
            }
        }
        let l = chol(&sp);
        let mut pc = Matrix::zeros(k, k);
        for x in 0..k {
            for y in 0..k {
                pc[(order[x], order[y])] = l[(x, y)];
            }
        }
        let theta: Vec<Matrix> = phi.iter().map(|f| mul(f, &pc)).collect();
        let psr = if self.kind == Kind::Svar {
            let o = self.off_ab();
            let am = Matrix::from_rows(&(0..k).map(|i| (0..k).map(|j| th[o + j * k + i]).collect()).collect::<Vec<_>>());
            let bm = Matrix::from_rows(&(0..k).map(|i| (0..k).map(|j| th[o + k * k + j * k + i]).collect()).collect::<Vec<_>>());
            Some(mul(&inv(&am).unwrap_or_else(|| Matrix::identity(k)), &bm))
        } else {
            None
        };
        let thsr: Vec<Matrix> = psr.as_ref().map(|ps| phi.iter().map(|f| mul(f, ps)).collect()).unwrap_or_default();
        let n = h + 1;
        let mut out = Vec::new();
        let cum = |f: &dyn Fn(usize, usize, usize) -> f64, nimp: usize, out: &mut Vec<f64>, cumulative: bool| {
            for i in 0..nimp {
                for j in 0..k {
                    let mut acc = 0.0;
                    for s in 0..n {
                        let v = f(i, j, s);
                        acc = if cumulative { acc + v } else { v };
                        out.push(acc);
                    }
                }
            }
        };
        let dm = |i: usize, j: usize, s: usize| -> f64 { (0..=s.min(self.s)).map(|l| (0..k).map(|q| phi[s - l][(j, q)] * bx[l][(q, i)]).sum::<f64>()).sum() };
        let fevd = |ths: &[Matrix], i: usize, j: usize, s: usize, structural: bool| -> f64 {
            if s == 0 {
                return 0.0;
            }
            let num: f64 = (0..s).map(|q| ths[q][(j, i)].powi(2)).sum();
            let den: f64 = if structural {
                (0..s).map(|q| (0..k).map(|c| ths[q][(j, c)].powi(2)).sum::<f64>()).sum()
            } else {
                (0..s).map(|q| (0..k).map(|a| (0..k).map(|b| phi[q][(j, a)] * sigma[(a, b)] * phi[q][(j, b)]).sum::<f64>()).sum::<f64>()).sum()
            };
            num / den
        };
        for st in self.avail() {
            match st {
                "irf" => cum(&|i, j, s| phi[s][(j, i)], k, &mut out, false),
                "oirf" => cum(&|i, j, s| theta[s][(j, i)], k, &mut out, false),
                "dm" => cum(&dm, r, &mut out, false),
                "cirf" => cum(&|i, j, s| phi[s][(j, i)], k, &mut out, true),
                "coirf" => cum(&|i, j, s| theta[s][(j, i)], k, &mut out, true),
                "cdm" => cum(&dm, r, &mut out, true),
                "fevd" => cum(&|i, j, s| fevd(&theta, i, j, s, false), k, &mut out, false),
                "sirf" => cum(&|i, j, s| thsr[s][(j, i)], k, &mut out, false),
                "csirf" => cum(&|i, j, s| thsr[s][(j, i)], k, &mut out, true),
                "sfevd" => cum(&|i, j, s| fevd(&thsr, i, j, s, true), k, &mut out, false),
                _ => {}
            }
        }
        out
    }
}

/// Exogenous regressors grouped as (base variable, lag).
fn exog_structure(exog: &[String]) -> (Vec<String>, Vec<(usize, usize)>) {
    let mut bases: Vec<String> = Vec::new();
    let mut pos = Vec::new();
    for x in exog {
        let (lag, base) = match x.rsplit_once('.') {
            Some((op, v)) if op.starts_with('L') && op[1..].chars().all(|c| c.is_ascii_digit()) => (op[1..].parse().unwrap_or(1), v.to_string()),
            _ => (0usize, x.clone()),
        };
        let bi = match bases.iter().position(|b| *b == base) {
            Some(i) => i,
            None => {
                bases.push(base);
                bases.len() - 1
            }
        };
        pos.push((bi, lag));
    }
    (bases, pos)
}

fn model_from_e(e: &Results) -> Result<Model, String> {
    let cmd = e.macros.get("cmd").cloned().unwrap_or_default();
    match cmd.as_str() {
        "var" | "svar" => {
            let spec = tsvar::spec_from_e(e)?;
            let k = spec.endog.len();
            let p = spec.maxlag().max(1);
            let svar = cmd == "svar";
            let b = e.matrices.get(if svar { "b_var" } else { "b" }).cloned().ok_or_else(|| rc_err(301, "e(b) not found"))?;
            let v = e.matrices.get(if svar { "V_var" } else { "V" }).cloned().ok_or_else(|| rc_err(301, "e(V) not found"))?;
            let sigma = e.matrices.get("Sigma").cloned().ok_or_else(|| rc_err(301, "e(Sigma) not found"))?;
            let m = spec.regnames().len();
            let nl = spec.lags.len();
            let (bases, pos) = exog_structure(&spec.exog);
            let s = pos.iter().map(|x| x.1).max().unwrap_or(0);
            let mut model = Model { kind: if svar { Kind::Svar } else { Kind::Var }, k, p, exog: bases.clone(), s, theta: Vec::new(), vtheta: None, endog: spec.endog.clone(), t: e.get("N").unwrap_or(0.0), lags: spec.lags.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(" ") };
            let n = model.n_theta();
            let mut th = vec![0.0; n];
            // map θ index -> e(b) index
            let mut map: Vec<Option<usize>> = vec![None; n];
            for i in 0..k {
                for j in 0..k {
                    for (li, &l) in spec.lags.iter().enumerate() {
                        let ti = (l - 1) * k * k + i * k + j;
                        let bi = i * m + j * nl + li;
                        th[ti] = b.data[bi];
                        map[ti] = Some(bi);
                    }
                }
                for (xi, &(base, lag)) in pos.iter().enumerate() {
                    let ti = model.off_b() + lag * k * bases.len() + i * bases.len() + base;
                    let bi = i * m + k * nl + xi;
                    th[ti] = b.data[bi];
                    map[ti] = Some(bi);
                }
            }
            let vs = vech(&sigma);
            let os = model.off_sig();
            th[os..os + vs.len()].copy_from_slice(&vs);
            let mut vt = Matrix::zeros(n, n);
            for a in 0..n {
                for c in 0..n {
                    if let (Some(x), Some(y)) = (map[a], map[c]) {
                        vt[(a, c)] = v[(x, y)];
                    }
                }
            }
            let vsig = crate::tsla::vech_sigma_vce(&sigma, model.t);
            for a in 0..vs.len() {
                for c in 0..vs.len() {
                    vt[(os + a, os + c)] = vsig[(a, c)];
                }
            }
            if svar {
                let am = e.matrices.get("A").cloned().ok_or_else(|| rc_err(301, "e(A) not found"))?;
                let bm = e.matrices.get("B").cloned().ok_or_else(|| rc_err(301, "e(B) not found"))?;
                let vab = e.matrices.get("V").cloned().ok_or_else(|| rc_err(301, "e(V) not found"))?;
                let o = model.off_ab();
                for j in 0..k {
                    for i in 0..k {
                        th[o + j * k + i] = am[(i, j)];
                        th[o + k * k + j * k + i] = bm[(i, j)];
                    }
                }
                for a in 0..2 * k * k {
                    for c in 0..2 * k * k {
                        vt[(o + a, o + c)] = vab[(a, c)];
                    }
                }
            }
            model.theta = th;
            model.vtheta = Some(vt);
            Ok(model)
        }
        "vec" => {
            let rep = crate::tsvec::rep_from_e(e)?;
            let k = rep.endog.len();
            let p = rep.a.len();
            let mut model = Model { kind: Kind::Vec, k, p, exog: Vec::new(), s: 0, theta: Vec::new(), vtheta: None, endog: rep.endog.clone(), t: rep.t, lags: (1..=p).map(|l| l.to_string()).collect::<Vec<_>>().join(" ") };
            let mut th = vec![0.0; model.n_theta()];
            for (l, a) in rep.a.iter().enumerate() {
                for i in 0..k {
                    for j in 0..k {
                        th[l * k * k + i * k + j] = a[(i, j)];
                    }
                }
            }
            let vs = vech(&rep.omega);
            let os = model.off_sig();
            th[os..os + vs.len()].copy_from_slice(&vs);
            model.theta = th;
            Ok(model)
        }
        _ => Err(rc_err(301, "irf create requires var, svar or vec estimation results")),
    }
}

fn find_name(list: &[String], name: &str) -> Option<usize> {
    let n = name.trim();
    list.iter().position(|x| x.eq_ignore_ascii_case(n) || tsvar::eqname(x).eq_ignore_ascii_case(n) || tsvar::eqname(x).eq_ignore_ascii_case(&tsvar::eqname(n)))
}

/// Bootstrap standard errors of all statistics after `var`.
fn bootstrap(ds: &DataSet, st: &Store, e: &Results, model: &Model, h: usize, order: &[usize], reps: usize, parametric: bool) -> Result<Vec<f64>, String> {
    let f = tsvar::fit_from_e(ds, st, e)?;
    if f.data.times.windows(2).any(|w| w[1] - w[0] > 1.0 + 1e-9) {
        return Err(rc_err(498, "bootstrap not allowed with gaps in the data"));
    }
    let k = f.k;
    let t = f.t;
    let nl = f.spec.lags.len();
    let lags = f.spec.lags.clone();
    let chol_s = chol(&f.sigma);
    let (bases, pos) = exog_structure(&f.spec.exog);
    let _ = bases;
    let mut sum: Vec<f64> = Vec::new();
    let mut sumsq: Vec<f64> = Vec::new();
    let mut rng = st.rng.borrow_mut();
    for _ in 0..reps {
        let mut ys: Vec<Vec<f64>> = Vec::with_capacity(t);
        let mut xs: Vec<Vec<f64>> = Vec::with_capacity(t);
        for i in 0..t {
            let mut x = f.data.x[i].clone();
            for j in 0..k {
                for (li, &l) in lags.iter().enumerate() {
                    if i >= l {
                        x[j * nl + li] = ys[i - l][j];
                    }
                }
            }
            let u: Vec<f64> = if parametric {
                let z: Vec<f64> = (0..k).map(|_| rng.normal()).collect();
                (0..k).map(|a| (0..k).map(|b| chol_s[(a, b)] * z[b]).sum()).collect()
            } else {
                let r = ((rng.uniform() * t as f64) as usize).min(t - 1);
                f.resid[r].clone()
            };
            let y: Vec<f64> = (0..k).map(|j| (0..f.m).map(|a| f.coef[j][a] * x[a]).sum::<f64>() + u[j]).collect();
            ys.push(y);
            xs.push(x);
        }
        let (coef, _, resid) = tsvar::ols_system(&ys, &xs);
        let sig = tsvar::cross(&resid, if f.spec.dfk { (t - f.m) as f64 } else { t as f64 });
        let mut th = model.theta.clone();
        for i in 0..k {
            for j in 0..k {
                for (li, &l) in lags.iter().enumerate() {
                    th[(l - 1) * k * k + i * k + j] = coef[i][j * nl + li];
                }
            }
            for (xi, &(base, lag)) in pos.iter().enumerate() {
                th[model.off_b() + lag * k * model.exog.len() + i * model.exog.len() + base] = coef[i][k * nl + xi];
            }
        }
        let vs = vech(&sig);
        let os = model.off_sig();
        th[os..os + vs.len()].copy_from_slice(&vs);
        let out = model.compute(&th, h, order);
        if sum.is_empty() {
            sum = vec![0.0; out.len()];
            sumsq = vec![0.0; out.len()];
        }
        for (q, v) in out.iter().enumerate() {
            sum[q] += v;
            sumsq[q] += v * v;
        }
    }
    let r = reps as f64;
    Ok(sum.iter().zip(&sumsq).map(|(s, q)| ((q - s * s / r) / (r - 1.0)).max(0.0).sqrt()).collect())
}

fn create(ds: &DataSet, st: &mut Store, cmd: &Command, name: &str) -> Result<String, String> {
    let mut msgs = Vec::new();
    if name.is_empty() || name.len() > 15 {
        return Err(rc_err(198, "irfname required (at most 15 characters)"));
    }
    let e = tsvar::est_results(st, cmd)?;
    let model = model_from_e(&e)?;
    let path = file_for(st, cmd, &mut msgs)?;
    let mut file = IrfFile::load(&path)?;
    if file.get(name).is_some() && !cmd.has_opt("replace", 3) {
        return Err(rc_err(110, format!("irfname {name} already exists in {path}")));
    }
    let h = util::opt_num(cmd, "step", 3)?.unwrap_or(8.0);
    if h < 1.0 {
        return Err(rc_err(198, "step() must be positive"));
    }
    let h = h as usize;
    let order: Vec<usize> = match cmd.opt("order", 3) {
        Some(o) => {
            let names = split_top_level_words(o);
            let idx: Vec<usize> = names.iter().map(|n| find_name(&model.endog, n).ok_or_else(|| rc_err(198, format!("{n} is not an endogenous variable")))).collect::<Result<_, _>>()?;
            let mut chk = idx.clone();
            chk.sort();
            chk.dedup();
            if chk.len() != model.k || idx.len() != model.k {
                return Err(rc_err(198, "order() must contain each endogenous variable once"));
            }
            idx
        }
        None => (0..model.k).collect(),
    };
    let point = model.compute(&model.theta, h, &order);
    let bs = cmd.has_opt("bs", 2) && !cmd.has_opt("bsp", 3);
    let bsp = cmd.has_opt("bsp", 3);
    let reps = util::opt_num(cmd, "reps", 3)?.unwrap_or(200.0) as usize;
    let (se, stderror): (Option<Vec<f64>>, &str) = if cmd.has_opt("nose", 4) {
        (None, "none")
    } else if bs || bsp {
        if model.kind != Kind::Var {
            return Err(rc_err(198, "bootstrap standard errors are available after var only"));
        }
        if reps < 2 {
            return Err(rc_err(198, "reps() must be at least 2"));
        }
        (Some(bootstrap(ds, st, &e, &model, h, &order, reps, bsp)?), if bsp { "bsp" } else { "bs" })
    } else {
        match &model.vtheta {
            Some(v) => {
                let f = |x: &[f64]| model.compute(x, h, &order);
                let j = crate::tsla::jacobian(&f, &model.theta);
                (Some(crate::tsla::delta_var(&j, v).iter().map(|x| x.sqrt()).collect()), "asymptotic")
            }
            None => (None, "none"),
        }
    };
    let k = model.k;
    let mut res = IrfResult { name: name.to_string(), endog: model.endog.clone(), exog: model.exog.clone(), step: h, ..Default::default() };
    let mut off = 0;
    for stat in model.avail() {
        let nimp = if stat.ends_with("dm") { model.exog.len() } else { k };
        let len = nimp * k * (h + 1);
        let mut vals = point[off..off + len].to_vec();
        // dm series are stored on an endog-sized grid: impulse index over exog
        if nimp != k {
            vals.resize(nimp * k * (h + 1), SYSMISS);
        }
        res.vals.insert(stat.to_string(), vals);
        if let Some(se) = &se {
            res.vals.insert(format!("std{stat}"), se[off..off + len].to_vec());
        }
        off += len;
    }
    let model_name = match model.kind {
        Kind::Var => "var",
        Kind::Svar => "sr var",
        Kind::Vec => "vec",
    };
    let meta: Vec<(&str, String)> = vec![
        ("model", model_name.into()),
        ("order", order.iter().map(|&i| model.endog[i].clone()).collect::<Vec<_>>().join(" ")),
        ("exog", e.macros.get("exog").or_else(|| e.macros.get("exog_var")).cloned().unwrap_or_default()),
        ("exogvars", model.exog.join(" ")),
        ("constant", if e.macros.get("nocons").or_else(|| e.macros.get("nocons_var")).is_some() { "noconstant".into() } else { "constant".into() }),
        ("lags", model.lags.clone()),
        ("tmin", fmt_g(e.get("tmin").unwrap_or(SYSMISS), 12)),
        ("tmax", fmt_g(e.get("tmax").unwrap_or(SYSMISS), 12)),
        ("timevar", e.macros.get("timevar").or_else(|| e.macros.get("tvar")).cloned().unwrap_or_default()),
        ("tsfmt", e.macros.get("tsfmt").cloned().unwrap_or_default()),
        ("step", h.to_string()),
        ("stderror", stderror.into()),
        ("reps", if bs || bsp { reps.to_string() } else { String::new() }),
        ("version", "1.1".into()),
        ("rank", e.get("k_ce").map(|x| fmt_g(x, 6)).unwrap_or_default()),
        ("trend", e.macros.get("trend").cloned().unwrap_or_default()),
        ("endog", model.endog.join(" ")),
    ];
    res.meta = meta.into_iter().map(|(a, b)| (a.to_string(), b)).collect();
    file.results.retain(|r| r.name != name);
    file.results.push(res);
    file.save(&path)?;
    msgs.push(format!("(file {path} updated)"));
    Ok(msgs.join("\n"))
}

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

fn fmt_irf(v: f64) -> String {
    if is_missing(v) {
        return ".".into();
    }
    if v == 0.0 {
        return "0".into();
    }
    if v.abs() >= 1e-6 && v.abs() < 1e5 {
        let s = format!("{v:.6}");
        let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
        let s = if let Some(r) = s.strip_prefix("0.") { format!(".{r}") } else if let Some(r) = s.strip_prefix("-0.") { format!("-.{r}") } else { s };
        if s.len() <= 9 && s != "-0" && !s.is_empty() {
            return s;
        }
    }
    fmt_g(v, 8)
}

/// One key of a table: an IRF result, impulse, response and statistics.
struct Key<'a> {
    res: &'a IrfResult,
    imp: usize,
    resp: usize,
    stats: Vec<String>,
}

struct TabOpts {
    ci: bool,
    se: bool,
    level: f64,
    step: Option<usize>,
}

fn columns<'a>(keys: &'a [Key<'a>], o: &TabOpts) -> Vec<(usize, String, Box<dyn Fn(usize) -> f64 + 'a>)> {
    let z = dist::invnormal(1.0 - (1.0 - o.level / 100.0) / 2.0);
    let mut cols: Vec<(usize, String, Box<dyn Fn(usize) -> f64 + 'a>)> = Vec::new();
    for (ki, key) in keys.iter().enumerate() {
        for s in &key.stats {
            let (res, i, j) = (key.res, key.imp, key.resp);
            let s1 = s.clone();
            cols.push((ki, s.clone(), Box::new(move |h| res.get(&s1, i, j, h))));
            let has_se = res.has(&format!("std{s}"));
            if o.ci && has_se {
                let s2 = s.clone();
                cols.push((ki, "Lower".into(), Box::new(move |h| res.get(&s2, i, j, h) - z * res.get(&format!("std{s2}"), i, j, h))));
                let s3 = s.clone();
                cols.push((ki, "Upper".into(), Box::new(move |h| res.get(&s3, i, j, h) + z * res.get(&format!("std{s3}"), i, j, h))));
            }
            if o.se && has_se {
                let s4 = s.clone();
                cols.push((ki, "S.E.".into(), Box::new(move |h| res.get(&format!("std{s4}"), i, j, h))));
            }
        }
    }
    cols
}

fn render_table(keys: &[Key], o: &TabOpts, title: Option<&str>, key_offset: usize) -> Vec<String> {
    let cols = columns(keys, o);
    let maxstep = keys.iter().map(|k| k.res.step).max().unwrap_or(0);
    let nsteps = o.step.map_or(maxstep, |s| s.min(maxstep));
    let mut out = Vec::new();
    if let Some(t) = title {
        out.push(String::new());
        out.push(t.to_string());
    }
    // blocks of at most 5 columns, keeping keys together when possible
    let mut blocks: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut ki = 0;
    while ki < keys.len() {
        let kc: Vec<usize> = cols.iter().enumerate().filter(|c| c.1 .0 == ki).map(|c| c.0).collect();
        if !cur.is_empty() && cur.len() + kc.len() > 5 {
            blocks.push(std::mem::take(&mut cur));
        }
        for c in kc {
            if cur.len() == 5 {
                blocks.push(std::mem::take(&mut cur));
            }
            cur.push(c);
        }
        ki += 1;
    }
    if !cur.is_empty() {
        blocks.push(cur);
    }
    for b in blocks {
        let w = 8 + 1 + b.len() * 13;
        out.push(String::new());
        out.push(format!("+{}+", "-".repeat(w)));
        out.push(format!("|{:8}|{}|", "", b.iter().map(|&c| format!("{:>11}  ", format!("({})", cols[c].0 + 1 + key_offset))).collect::<String>()));
        out.push(format!("|{:^8}|{}|", "Step", b.iter().map(|&c| format!("{:>11}  ", cols[c].1)).collect::<String>()));
        out.push(format!("|{}+{}|", "-".repeat(8), "-".repeat(w - 8 - 1 + 1)));
        for h in 0..=nsteps {
            out.push(format!("|{:<8}|{}|", h, b.iter().map(|&c| format!("{:>11}  ", fmt_irf((cols[c].2)(h)))).collect::<String>()));
        }
        out.push(format!("+{}+", "-".repeat(w)));
    }
    if o.ci && cols.iter().any(|c| c.1 == "Lower") {
        out.push(format!("{}% lower and upper bounds reported.", fmt_g(o.level, 4)));
    }
    for (i, k) in keys.iter().enumerate() {
        let imps = k.res.impulses(k.stats.first().map(String::as_str).unwrap_or("irf"));
        out.push(format!("({}) irfname = {}, impulse = {}, and response = {}.", i + 1 + key_offset, k.res.name, imps.get(k.imp).cloned().unwrap_or_default(), k.res.endog.get(k.resp).cloned().unwrap_or_default()));
    }
    out
}

fn tab_opts(cmd: &Command, st: &Store) -> Result<TabOpts, String> {
    Ok(TabOpts {
        ci: !cmd.has_opt("noci", 4),
        se: cmd.has_opt("stderror", 3),
        level: util::opt_num(cmd, "level", 3)?.unwrap_or(st.level),
        step: util::opt_num(cmd, "step", 3)?.map(|x| x as usize),
    })
}

fn names_opt(cmd: &Command, name: &str, min: usize) -> Option<Vec<String>> {
    cmd.opt(name, min).map(|s| split_top_level_words(s))
}

/// Keys for (irfnames × impulses × responses) with the requested statistics.
fn select<'a>(file: &'a IrfFile, cmd: &Command, stats: &[String], nostructural: bool) -> Result<Vec<Key<'a>>, String> {
    let irfs: Vec<&IrfResult> = match names_opt(cmd, "irf", 3) {
        Some(ns) => ns.iter().map(|n| file.get(n).ok_or_else(|| rc_err(111, format!("irfname {n} not found")))).collect::<Result<_, _>>()?,
        None => file.results.iter().collect(),
    };
    if irfs.is_empty() {
        return Err(rc_err(111, "no IRF results in the active file"));
    }
    let imps = names_opt(cmd, "impulse", 2);
    let resps = names_opt(cmd, "response", 2);
    let mut keys = Vec::new();
    for res in irfs {
        let st: Vec<String> = if stats.is_empty() {
            STATS.iter().filter(|s| res.has(s) && !(nostructural && matches!(**s, "sirf" | "csirf" | "sfevd")) && !s.ends_with("dm")).map(|s| s.to_string()).collect()
        } else {
            stats.iter().filter(|s| res.has(s)).cloned().collect()
        };
        if st.is_empty() {
            continue;
        }
        let dm = st.iter().all(|s| s.ends_with("dm"));
        let implist = if dm { &res.exog } else { &res.endog };
        let iidx: Vec<usize> = match &imps {
            Some(v) => v.iter().filter_map(|n| find_name(implist, n)).collect(),
            None => (0..implist.len()).collect(),
        };
        let ridx: Vec<usize> = match &resps {
            Some(v) => v.iter().filter_map(|n| find_name(&res.endog, n)).collect(),
            None => (0..res.endog.len()).collect(),
        };
        for &i in &iidx {
            for &j in &ridx {
                keys.push(Key { res, imp: i, resp: j, stats: st.clone() });
            }
        }
    }
    if keys.is_empty() {
        return Err(rc_err(111, "no IRF results match the specification"));
    }
    Ok(keys)
}

fn load_for(st: &mut Store, cmd: &Command) -> Result<(String, IrfFile), String> {
    let mut msgs = Vec::new();
    let path = file_for(st, cmd, &mut msgs)?;
    Ok((path.clone(), IrfFile::load(&path)?))
}

fn table(st: &mut Store, cmd: &Command, stats: Vec<String>) -> Result<String, String> {
    for s in &stats {
        if !STATS.contains(&s.as_str()) {
            return Err(rc_err(198, format!("{s} invalid statistic")));
        }
    }
    let (_, file) = load_for(st, cmd)?;
    let keys = select(&file, cmd, &stats, cmd.has_opt("nostructural", 5))?;
    let o = tab_opts(cmd, st)?;
    let mut out = Vec::new();
    if cmd.has_opt("individual", 3) {
        let mut names: Vec<&str> = keys.iter().map(|k| k.res.name.as_str()).collect();
        names.dedup();
        for n in names {
            let sub: Vec<Key> = keys.iter().filter(|k| k.res.name == n).map(|k| Key { res: k.res, imp: k.imp, resp: k.resp, stats: k.stats.clone() }).collect();
            out.extend(render_table(&sub, &o, Some(&format!("Results from {n}")), 0));
        }
    } else {
        let mut names: Vec<&str> = keys.iter().map(|k| k.res.name.as_str()).collect();
        names.dedup();
        let title = cmd.opt("title", 2).map(util::unquote).or_else(|| if names.len() == 1 { Some(format!("Results from {}", names[0])) } else { None });
        out.extend(render_table(&keys, &o, title.as_deref(), 0));
    }
    let r = st.begin_r();
    r.set("k", keys.len() as f64);
    for (i, k) in keys.iter().enumerate() {
        r.set_str(&format!("key{}", i + 1), format!("{} {} {}", k.res.name, k.res.endog.get(k.imp).cloned().unwrap_or_default(), k.res.endog.get(k.resp).cloned().unwrap_or_default()));
    }
    Ok(out.join("\n"))
}

/// Parenthesised specifications `(irfname impulse response stat [, opts])`.
fn tuples(text: &str) -> Result<Vec<(Vec<String>, String)>, String> {
    let mut out = Vec::new();
    let mut depth = 0;
    let mut cur = String::new();
    for c in text.chars() {
        match c {
            '(' => {
                if depth > 0 {
                    cur.push(c);
                }
                depth += 1;
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    let (main, opts) = match cur.split_once(',') {
                        Some((a, b)) => (a.to_string(), b.to_string()),
                        None => (cur.clone(), String::new()),
                    };
                    out.push((main.split_whitespace().map(String::from).collect(), opts));
                    cur.clear();
                } else {
                    cur.push(c);
                }
            }
            _ if depth > 0 => cur.push(c),
            _ => {}
        }
    }
    if out.is_empty() {
        return Err(rc_err(198, "specification (irfname impulse response stat) required"));
    }
    Ok(out)
}

fn tuple_key<'a>(file: &'a IrfFile, t: &[String]) -> Result<Key<'a>, String> {
    if t.len() < 4 {
        return Err(rc_err(198, "each specification needs irfname, impulse, response and statistic"));
    }
    let res = file.get(&t[0]).ok_or_else(|| rc_err(111, format!("irfname {} not found", t[0])))?;
    let stats: Vec<String> = t[3..].to_vec();
    let dm = stats.iter().all(|s| s.ends_with("dm"));
    let imp = find_name(if dm { &res.exog } else { &res.endog }, &t[1]).ok_or_else(|| rc_err(111, format!("impulse {} not found", t[1])))?;
    let resp = find_name(&res.endog, &t[2]).ok_or_else(|| rc_err(111, format!("response {} not found", t[2])))?;
    for s in &stats {
        if !res.has(s) {
            return Err(rc_err(111, format!("{s} not available for {}", res.name)));
        }
    }
    Ok(Key { res, imp, resp, stats })
}

fn ctable(st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let (_, file) = load_for(st, cmd)?;
    let specs = tuples(rest)?;
    let o = tab_opts(cmd, st)?;
    let mut out = Vec::new();
    let mut keys = Vec::new();
    let mut opts = Vec::new();
    for (t, op) in &specs {
        keys.push(tuple_key(&file, t)?);
        opts.push(op.clone());
    }
    // per-specification noci/stderror
    let all_same = opts.iter().all(|x| x.trim().is_empty());
    if all_same {
        out.extend(render_table(&keys, &o, cmd.opt("title", 2).map(util::unquote).as_deref(), 0));
    } else {
        for (i, (k, op)) in keys.into_iter().zip(opts).enumerate() {
            let c = parse_line(&format!("x , {op}")).map_err(|e| rc_err(198, e))?;
            let oi = TabOpts { ci: o.ci && !c.has_opt("noci", 4), se: o.se || c.has_opt("stderror", 3), level: c.opt("level", 3).and_then(|x| x.trim().parse().ok()).unwrap_or(o.level), step: o.step };
            out.extend(render_table(&[k], &oi, None, i));
        }
    }
    Ok(out.join("\n"))
}

// ---------------------------------------------------------------------------
// Graphs
// ---------------------------------------------------------------------------

fn panel(keys: &[(Key, bool)], lstep: usize, ustep: Option<usize>, level: f64) -> Graph {
    let z = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
    let mut g = Graph::default();
    let mut lines = Vec::new();
    let mut cis = Vec::new();
    let pal = crate::graph::palette("");
    let nstats: usize = keys.iter().map(|k| k.0.stats.len()).sum();
    let mut idx = 0;
    for (k, ci) in keys {
        let hmax = ustep.map_or(k.res.step, |u| u.min(k.res.step));
        for s in &k.stats {
            let c = pal[idx % pal.len()];
            let col = (c.0, c.1, c.2);
            let pts: Vec<(f64, f64)> = (lstep..=hmax).map(|h| (h as f64, k.res.get(s, k.imp, k.resp, h))).filter(|p| !is_missing(p.1)).collect();
            let label = if keys.len() > 1 { format!("{}: {}", k.res.name, stat_label(s)) } else { stat_label(s).to_string() };
            if *ci {
                let has = k.res.has(&format!("std{s}"));
                let ranges: Vec<(f64, f64, f64)> = if has {
                    (lstep..=hmax)
                        .filter_map(|h| {
                            let v = k.res.get(s, k.imp, k.resp, h);
                            let se = k.res.get(&format!("std{s}"), k.imp, k.resp, h);
                            if is_missing(v) || is_missing(se) { None } else { Some((h as f64, v - z * se, v + z * se)) }
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                let cilab = if nstats > 1 { format!("{}% CI for {s}", fmt_g(level, 4)) } else { format!("{}% CI", fmt_g(level, 4)) };
                cis.push(Layer { kind: LayerKind::Range(RangeStyle::Area), ranges, color: Some(col), alpha: 0.3, label: cilab, ..Default::default() });
            }
            lines.push(Layer { kind: LayerKind::Line, points: pts, color: Some(col), label, ..Default::default() });
            idx += 1;
        }
    }
    g.layers.extend(cis);
    g.layers.extend(lines);
    g.xtitle = "Step".into();
    g.legend = true;
    g
}

fn graph_out(st: &mut Store, cmd: &Command, mut panels: Vec<(String, Graph)>) -> Result<String, String> {
    let mut g = if panels.len() == 1 {
        let (t, mut g) = panels.pop().unwrap();
        g.title = t;
        g
    } else {
        let n = panels.len();
        let cols = (n as f64).sqrt().ceil() as usize;
        let mut root = Graph::default();
        for (t, mut c) in panels {
            c.title = t;
            root.children.push(c);
        }
        root.cols = cols;
        root
    };
    crate::graph::finish(&mut g, cmd, st)
}

fn steps(cmd: &Command) -> Result<(usize, Option<usize>), String> {
    Ok((util::opt_num(cmd, "lstep", 2)?.unwrap_or(0.0).max(0.0) as usize, util::opt_num(cmd, "ustep", 2)?.map(|x| x.max(0.0) as usize)))
}

fn graph(st: &mut Store, cmd: &Command, stats: Vec<String>) -> Result<String, String> {
    if stats.is_empty() || stats.len() > 4 {
        return Err(rc_err(198, "one to four statistics required"));
    }
    for s in &stats {
        if !STATS.contains(&s.as_str()) {
            return Err(rc_err(198, format!("{s} invalid statistic")));
        }
    }
    let (_, file) = load_for(st, cmd)?;
    let irfs: Vec<&IrfResult> = match names_opt(cmd, "irf", 3) {
        Some(ns) => ns.iter().map(|n| file.get(n).ok_or_else(|| rc_err(111, format!("irfname {n} not found")))).collect::<Result<_, _>>()?,
        None => file.results.iter().collect(),
    };
    let imps = names_opt(cmd, "impulse", 2);
    let resps = names_opt(cmd, "response", 2);
    let level = util::opt_num(cmd, "level", 3)?.unwrap_or(st.level);
    let (lstep, ustep) = steps(cmd)?;
    let ci = !cmd.has_opt("noci", 4);
    let mut panels = Vec::new();
    for res in irfs {
        if !stats.iter().any(|s| res.has(s)) {
            continue;
        }
        let dm = stats.iter().all(|s| s.ends_with("dm"));
        let implist = if dm { &res.exog } else { &res.endog };
        let iidx: Vec<usize> = match &imps {
            Some(v) => v.iter().filter_map(|n| find_name(implist, n)).collect(),
            None => (0..implist.len()).collect(),
        };
        let ridx: Vec<usize> = match &resps {
            Some(v) => v.iter().filter_map(|n| find_name(&res.endog, n)).collect(),
            None => (0..res.endog.len()).collect(),
        };
        for &i in &iidx {
            for &j in &ridx {
                let key = Key { res, imp: i, resp: j, stats: stats.clone() };
                let p = panel(&[(key, ci)], lstep, ustep, level);
                panels.push((format!("{}, {}, {}", res.name, implist[i], res.endog[j]), p));
            }
        }
    }
    if panels.is_empty() {
        return Err(rc_err(111, "no IRF results match the specification"));
    }
    graph_out(st, cmd, panels)
}

fn ograph(st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let (_, file) = load_for(st, cmd)?;
    let specs = tuples(rest)?;
    let level = util::opt_num(cmd, "level", 3)?.unwrap_or(st.level);
    let (lstep, ustep) = steps(cmd)?;
    let mut keys = Vec::new();
    for (t, op) in &specs {
        let ci = op.split(|c: char| !c.is_alphanumeric()).any(|w| w == "ci") || cmd.has_opt("ci", 2);
        keys.push((tuple_key(&file, t)?, ci));
    }
    let mut g = panel(&keys, lstep, ustep, level);
    // labels identify each specification
    let nci = g.layers.iter().filter(|l| matches!(l.kind, LayerKind::Range(_))).count();
    let mut li = nci;
    for (k, _) in &keys {
        for s in &k.stats {
            if let Some(l) = g.layers.get_mut(li) {
                l.label = format!("{}: {} of {} -> {}", k.res.name, s, k.res.impulses(s).get(k.imp).cloned().unwrap_or_default(), k.res.endog[k.resp]);
            }
            li += 1;
        }
    }
    crate::graph::finish(&mut g, cmd, st)
}

fn cgraph(st: &mut Store, cmd: &Command, rest: &str) -> Result<String, String> {
    let (_, file) = load_for(st, cmd)?;
    let specs = tuples(rest)?;
    let level = util::opt_num(cmd, "level", 3)?.unwrap_or(st.level);
    let (lstep, ustep) = steps(cmd)?;
    let mut panels = Vec::new();
    for (t, op) in &specs {
        let ci = !op.contains("noci") && !cmd.has_opt("noci", 4);
        let key = tuple_key(&file, t)?;
        let title = format!("{}: {}, {}", key.res.name, key.res.impulses(&key.stats[0]).get(key.imp).cloned().unwrap_or_default(), key.res.endog[key.resp]);
        panels.push((title, panel(&[(key, ci)], lstep, ustep, level)));
    }
    graph_out(st, cmd, panels)
}

// ---------------------------------------------------------------------------
// describe, set, drop, rename
// ---------------------------------------------------------------------------

fn describe(st: &mut Store, cmd: &Command, names: &[String]) -> Result<String, String> {
    let path = match cmd.opt("using", 2) {
        Some(u) => irf_path(u),
        None => {
            let mut msgs = Vec::new();
            file_for(st, cmd, &mut msgs)?
        }
    };
    let path = if let Some(u) = &cmd.using { irf_path(u) } else { path };
    let file = IrfFile::load(&path)?;
    let mut out = vec![String::new(), format!("Contains irf results from {path}{}", if file.created.is_empty() { String::new() } else { format!(" (dated {})", file.created) }), String::new()];
    out.push(format!("{:<15}{:<10}{}", "irfname", "model", "endogenous variables and order (*)"));
    out.push("-".repeat(70));
    let sel: Vec<&IrfResult> = file.results.iter().filter(|r| names.is_empty() || names.contains(&r.name)).collect();
    for r in &sel {
        out.push(format!("{:<15}{:<10}{}", r.name, r.meta("model"), r.meta("order")));
    }
    out.push("-".repeat(70));
    out.push("(*) order is relevant only when model is var".into());
    if cmd.has_opt("detail", 1) {
        for r in &sel {
            out.push(String::new());
            out.push(format!("Estimation specification for {}", r.name));
            for (k, v) in &r.meta {
                if !v.is_empty() {
                    out.push(format!("  {k:<12}: {v}"));
                }
            }
        }
    }
    let rr = st.begin_r();
    rr.set_str("irfnames", file.results.iter().map(|r| r.name.clone()).collect::<Vec<_>>().join(" "));
    rr.set_str("version", "1.1");
    for r in &file.results {
        for (k, v) in &r.meta {
            rr.set_str(&format!("{}_{k}", r.name), v.clone());
        }
    }
    Ok(out.join("\n"))
}

pub struct Irf;

impl CommandHandler for Irf {
    fn name(&self) -> &'static str {
        "irf"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (sub, rest) = match main.split_once(char::is_whitespace) {
            Some((a, b)) => (a.to_lowercase(), b.trim().to_string()),
            None => (main.to_lowercase(), String::new()),
        };
        let words: Vec<String> = rest.split_whitespace().map(|w| w.to_lowercase()).collect();
        match sub.as_str() {
            "create" | "cr" | "cre" | "crea" | "creat" => create(ds, st, cmd, rest.trim()),
            "set" => {
                if cmd.has_opt("clear", 5) {
                    st.globals.remove("S_vrffile");
                    return Ok(String::new());
                }
                if rest.is_empty() {
                    return Ok(match active_file(st) {
                        Some(f) => format!("(file {f} is active)"),
                        None => "no irf file active".into(),
                    });
                }
                let mut msgs = Vec::new();
                let spec = if cmd.has_opt("replace", 3) { format!("{rest}, replace") } else { rest.clone() };
                set_file(st, &spec, &mut msgs)?;
                Ok(msgs.join("\n"))
            }
            "table" | "tab" | "tabl" | "ta" => table(st, cmd, words),
            "ctable" | "ctab" | "ctabl" | "ct" => ctable(st, cmd, &rest),
            "graph" | "gr" | "gra" | "grap" => graph(st, cmd, words),
            "ograph" | "og" | "ogr" | "ogra" | "ograp" => ograph(st, cmd, &rest),
            "cgraph" | "cg" | "cgr" | "cgra" | "cgrap" => cgraph(st, cmd, &rest),
            "describe" | "d" | "de" | "des" | "desc" | "descr" | "descri" | "describ" => describe(st, cmd, &rest.split_whitespace().map(String::from).collect::<Vec<_>>()),
            "drop" => {
                let (path, mut file) = load_for(st, cmd)?;
                let names: Vec<String> = rest.split_whitespace().map(String::from).collect();
                let before = file.results.len();
                file.results.retain(|r| !names.contains(&r.name));
                file.save(&path)?;
                Ok(format!("({} irf result{} dropped)", before - file.results.len(), if before - file.results.len() == 1 { "" } else { "s" }))
            }
            "rename" | "ren" => {
                let w: Vec<&str> = rest.split_whitespace().collect();
                if w.len() != 2 {
                    return Err(rc_err(198, "syntax is irf rename oldname newname"));
                }
                let (path, mut file) = load_for(st, cmd)?;
                if file.get(w[1]).is_some() {
                    return Err(rc_err(110, format!("irfname {} already exists", w[1])));
                }
                let r = file.results.iter_mut().find(|r| r.name == w[0]).ok_or_else(|| rc_err(111, format!("irfname {} not found", w[0])))?;
                r.name = w[1].to_string();
                file.save(&path)?;
                Ok(format!("({} renamed to {})", w[0], w[1]))
            }
            "add" => {
                let (path, mut file) = load_for(st, cmd)?;
                let src = match (cmd.opt("using", 1), &cmd.using) {
                    (Some(u), _) => irf_path(u),
                    (None, Some(u)) => irf_path(u),
                    _ => return Err(rc_err(100, "using() required")),
                };
                let other = IrfFile::load(&src)?;
                // tokens: `name`, `_all`, or `newname = oldname`
                let spaced = rest.replace('=', " = ");
                let toks: Vec<&str> = spaced.split_whitespace().collect();
                let mut pairs: Vec<(String, String)> = Vec::new(); // (old, new)
                let mut i = 0;
                while i < toks.len() {
                    if i + 2 < toks.len() && toks[i + 1] == "=" {
                        pairs.push((toks[i + 2].to_string(), toks[i].to_string()));
                        i += 3;
                    } else {
                        pairs.push((toks[i].to_string(), toks[i].to_string()));
                        i += 1;
                    }
                }
                let all = pairs.is_empty() || pairs.iter().any(|(o, _)| o == "_all");
                let mut n = 0;
                for (old, _) in &pairs {
                    if old != "_all" && !other.results.iter().any(|r| &r.name == old) {
                        return Err(rc_err(111, format!("irfname {old} not found in {src}")));
                    }
                }
                for r in other.results {
                    let target = if all { Some(r.name.clone()) } else { pairs.iter().find(|(o, _)| *o == r.name).map(|(_, nw)| nw.clone()) };
                    if let Some(nm) = target {
                        let mut r = r;
                        r.name = nm;
                        file.results.retain(|q| q.name != r.name);
                        file.results.push(r);
                        n += 1;
                    }
                }
                file.save(&path)?;
                Ok(format!("({n} irf result{} added)", if n == 1 { "" } else { "s" }))
            }
            "" => Err(rc_err(198, "subcommand required")),
            other => Err(rc_err(199, format!("irf {other} is unrecognized"))),
        }
    }
}

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![("irf", Box::new(Irf)), ("varirf", Box::new(Irf))]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_roundtrip() {
        let mut r = IrfResult { name: "a".into(), endog: vec!["x".into(), "y".into()], step: 1, ..Default::default() };
        r.vals.insert("irf".into(), vec![1.0, 0.5, 0.0, 0.25, 0.0, SYSMISS, 1.0, 0.125]);
        r.meta.push(("model".into(), "var".into()));
        let f = IrfFile { results: vec![r], created: String::new() };
        let g = IrfFile::parse(&f.to_text()).unwrap();
        assert_eq!(g.results[0].get("irf", 0, 1, 1), 0.25);
        assert!(is_missing(g.results[0].get("irf", 1, 0, 1)));
        assert_eq!(g.results[0].meta("model"), "var");
        assert_eq!(fmt_irf(0.343150), ".34315");
        assert_eq!(fmt_irf(-0.000221), "-.000221");
    }
}
