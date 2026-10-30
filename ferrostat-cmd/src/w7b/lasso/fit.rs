//! Lasso fits as stored results: building the problem from a command
//! line, running a selection method over the lambda grid, and saving the
//! path in hidden e() macros so the postestimation commands (and stored
//! estimates) can replay it.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::core::{self, Family, Point, Problem};
use crate::est;
use crate::eval::rc_err;
use crate::store::{Results, Store};
use crate::util;
use crate::xmodels as xm;

/// One point of a stored path.
#[derive(Clone, Debug)]
pub struct Pt {
    pub alpha: f64,
    pub lambda: f64,
    pub b0: f64,
    pub b: Vec<f64>,
    /// mean in-sample deviance
    pub dev: f64,
    pub df: usize,
    pub cvm: f64,
    pub cvse: f64,
    pub bic: f64,
}

/// A stored lasso: its variables, penalty loadings and path.
#[derive(Clone, Debug)]
pub struct Lasso {
    pub family: Family,
    /// lasso, elasticnet or sqrtlasso
    pub kind: String,
    pub depvar: String,
    pub names: Vec<String>,
    pub always: Vec<bool>,
    pub loadings: Vec<f64>,
    /// cv, adaptive, plugin, bic or none
    pub selection: String,
    pub n: usize,
    pub nfolds: usize,
    /// index of the selected point
    pub sel: Option<usize>,
    /// null deviance (mean)
    pub dev0: f64,
    pub pts: Vec<Pt>,
    /// how the selected point was chosen (cv, serule, user, plugin, bic)
    pub how: String,
}

fn num(v: f64) -> String {
    if v.is_finite() {
        format!("{v:?}")
    } else {
        ".".into()
    }
}

fn parse_num(s: &str) -> f64 {
    if s == "." {
        f64::NAN
    } else {
        s.parse().unwrap_or(f64::NAN)
    }
}

impl Lasso {
    pub fn encode(&self) -> String {
        let mut out = vec![format!(
            "{} {} {} {} {} {} {} {} {}",
            self.family.name(),
            self.kind,
            self.depvar,
            self.selection,
            self.n,
            self.nfolds,
            self.sel.map_or(".".into(), |s| s.to_string()),
            num(self.dev0),
            if self.how.is_empty() { "-" } else { &self.how }
        )];
        out.push(self.names.join(" "));
        out.push(self.always.iter().map(|a| if *a { "1" } else { "0" }).collect::<Vec<_>>().join(" "));
        out.push(self.loadings.iter().map(|v| num(*v)).collect::<Vec<_>>().join(" "));
        for p in &self.pts {
            let mut f = vec![num(p.alpha), num(p.lambda), num(p.b0), num(p.dev), p.df.to_string(), num(p.cvm), num(p.cvse), num(p.bic)];
            f.extend(p.b.iter().map(|v| num(*v)));
            out.push(f.join(" "));
        }
        out.join("\n")
    }

    pub fn decode(s: &str) -> Option<Lasso> {
        let mut lines = s.split('\n');
        let h: Vec<&str> = lines.next()?.split(' ').collect();
        if h.len() < 9 {
            return None;
        }
        let names: Vec<String> = lines.next()?.split_whitespace().map(String::from).collect();
        let always: Vec<bool> = lines.next()?.split_whitespace().map(|v| v == "1").collect();
        let loadings: Vec<f64> = lines.next()?.split_whitespace().map(parse_num).collect();
        let mut pts = Vec::new();
        for l in lines {
            let f: Vec<&str> = l.split(' ').collect();
            if f.len() < 8 {
                continue;
            }
            pts.push(Pt {
                alpha: parse_num(f[0]),
                lambda: parse_num(f[1]),
                b0: parse_num(f[2]),
                dev: parse_num(f[3]),
                df: f[4].parse().unwrap_or(0),
                cvm: parse_num(f[5]),
                cvse: parse_num(f[6]),
                bic: parse_num(f[7]),
                b: f[8..].iter().map(|v| parse_num(v)).collect(),
            });
        }
        Some(Lasso {
            family: Family::parse(h[0])?,
            kind: h[1].into(),
            depvar: h[2].into(),
            selection: h[3].into(),
            n: h[4].parse().ok()?,
            nfolds: h[5].parse().unwrap_or(0),
            sel: h[6].parse().ok(),
            dev0: parse_num(h[7]),
            how: if h[8] == "-" { String::new() } else { h[8].into() },
            names,
            always,
            loadings,
            pts,
        })
    }

    pub fn selected(&self) -> Option<&Pt> {
        self.sel.and_then(|s| self.pts.get(s))
    }

    /// Names of the covariates with nonzero coefficients at point `k`.
    pub fn nonzero(&self, k: usize) -> Vec<usize> {
        self.pts.get(k).map(|p| (0..self.names.len()).filter(|&j| p.b[j] != 0.0).collect()).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// storing in e()
// ---------------------------------------------------------------------------

/// Store lassos with labels (the variable each one models).
pub fn save(e: &mut Results, lassos: &[(String, Lasso)]) {
    let labels: Vec<String> = lassos.iter().map(|(l, _)| l.clone()).collect();
    e.set_str("_lasso_list", labels.join(" "));
    for (k, (_, l)) in lassos.iter().enumerate() {
        e.set_str(&format!("_lasso{k}"), l.encode());
    }
}

/// All stored lassos of a result set.
pub fn load_all(e: &Results) -> Vec<(String, Lasso)> {
    let Some(list) = e.macros.get("_lasso_list") else { return Vec::new() };
    list.split_whitespace()
        .enumerate()
        .filter_map(|(k, l)| e.macros.get(&format!("_lasso{k}")).and_then(|s| Lasso::decode(s)).map(|x| (l.to_string(), x)))
        .collect()
}

/// The lasso picked by `for(name)` (default: the only/first one).
pub fn load_one(e: &Results, cmd: &Command) -> Result<(usize, Lasso), String> {
    let all = load_all(e);
    if all.is_empty() {
        return Err(rc_err(301, "last estimates not found; lasso results required"));
    }
    match cmd.opt("for", 3) {
        Some(f) => {
            let f = f.trim();
            all.into_iter().enumerate().find(|(_, (l, _))| l == f).map(|(k, (_, x))| (k, x)).ok_or_else(|| rc_err(198, format!("for({f}): no lasso for {f}")))
        }
        None => {
            if all.len() > 1 && e.macros.get("cmd").is_some_and(|c| c != "lasso" && c != "elasticnet" && c != "sqrtlasso") {
                return Err(rc_err(198, format!("option for() required; lassos available for: {}", all.iter().map(|(l, _)| l.as_str()).collect::<Vec<_>>().join(" "))));
            }
            Ok((0, all.into_iter().next().unwrap().1))
        }
    }
}

// ---------------------------------------------------------------------------
// data
// ---------------------------------------------------------------------------

/// Estimation data for one lasso: covariate columns (base levels of
/// factor variables removed) with always-included flags.
pub struct Data {
    pub names: Vec<String>,
    pub always: Vec<bool>,
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
    pub w: Vec<f64>,
    pub rows: Vec<usize>,
    pub mask: Vec<bool>,
}

/// Split `dep (always) others` (after an optional model word).
pub fn parse_main(main: &str) -> (String, String, String) {
    let groups = xm::paren_groups(main);
    let outside = xm::outside_parens(main);
    let (dep, rest) = xm::first_word(&outside);
    (dep, groups.join(" "), rest)
}

/// Build the data for depvar on `always` + `others` over the sample of
/// the command (if/in/weights), marking out `extra` variables too.
pub fn build(ds: &DataSet, st: &Store, cmd: &Command, dep: &str, always: &str, others: &str, extra: &[String]) -> Result<Data, String> {
    let indep = format!("{always} {others}");
    let mut d = est::build(ds, st, cmd, est::DesignSpec { depvar: Some(dep), indep: &indep, noconstant: true, weights: "fi", extra: extra.to_vec() })?;
    // lasso keeps collinear covariates: only base levels are dropped
    d.omitted = d.is_base.clone();
    let anames: Vec<String> = if always.trim().is_empty() {
        Vec::new()
    } else {
        est::build(ds, st, cmd, est::DesignSpec { depvar: None, indep: always, noconstant: true, weights: "fi", extra: Vec::new() })?.names
    };
    let mut names = Vec::new();
    let mut alw = Vec::new();
    let mut x = Vec::new();
    for j in 0..d.k() {
        if d.is_base[j] {
            continue;
        }
        if names.contains(&d.names[j]) {
            continue;
        }
        names.push(d.names[j].clone());
        alw.push(anames.contains(&d.names[j]));
        x.push(d.cols[j].clone());
    }
    if d.n() == 0 {
        return Err(rc_err(2000, "no observations"));
    }
    Ok(Data { names, always: alw, x, y: d.y.clone(), w: d.w.clone(), rows: d.rows.clone(), mask: d.mask.clone() })
}

/// Check and recode the outcome for the family.
pub fn outcome(fam: Family, y: &[f64], dep: &str) -> Result<Vec<f64>, String> {
    match fam {
        Family::Logit | Family::Probit => {
            let v: Vec<f64> = y.iter().map(|v| if *v != 0.0 { 1.0 } else { 0.0 }).collect();
            let s: f64 = v.iter().sum();
            if s == 0.0 || s == v.len() as f64 {
                return Err(rc_err(2000, format!("outcome {dep} does not vary")));
            }
            Ok(v)
        }
        Family::Poisson => {
            if y.iter().any(|v| *v < 0.0) {
                return Err(rc_err(459, format!("{dep} must be greater than or equal to zero")));
            }
            Ok(y.to_vec())
        }
        Family::Linear => Ok(y.to_vec()),
    }
}

// ---------------------------------------------------------------------------
// selection settings
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Sel {
    Cv,
    Adaptive,
    Plugin,
    Bic,
    None,
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub sel: Sel,
    pub folds: usize,
    pub serule: bool,
    pub steps: usize,
    pub power: f64,
    pub homoskedastic: bool,
    pub grid_n: usize,
    pub ratio: Option<f64>,
    pub lmin: Option<f64>,
    pub stop: f64,
    pub standardize: bool,
    pub alphas: Vec<f64>,
    pub sqrt: bool,
    pub lambdas: Option<Vec<f64>>,
}

fn sub_opt<'a>(opts: &'a [(String, Option<String>)], name: &str, min: usize) -> Option<Option<&'a str>> {
    opts.iter().find(|(o, _)| o.len() >= min && name.starts_with(o.to_lowercase().as_str())).map(|(_, v)| v.as_deref())
}

fn sub_num(opts: &[(String, Option<String>)], name: &str, min: usize) -> Result<Option<f64>, String> {
    match sub_opt(opts, name, min) {
        Some(Some(v)) => v.trim().parse::<f64>().map(Some).map_err(|_| rc_err(198, format!("invalid {name}()"))),
        Some(None) => Err(rc_err(198, format!("{name}() requires a value"))),
        None => Ok(None),
    }
}

/// Parse selection(), grid(), stop(), alpha() ... with `default` method.
pub fn settings(cmd: &Command, default: Sel) -> Result<Settings, String> {
    let mut s = Settings { sel: default, folds: 10, serule: false, steps: 2, power: 1.0, homoskedastic: false, grid_n: 100, ratio: None, lmin: None, stop: 1e-5, standardize: !cmd.has_opt("nostandardize", 7), alphas: vec![1.0], sqrt: false, lambdas: None };
    if let Some(sel) = cmd.opt("selection", 3) {
        let (m, rest) = match sel.find(',') {
            Some(p) => (sel[..p].trim().to_lowercase(), &sel[p + 1..]),
            None => (sel.trim().to_lowercase(), ""),
        };
        let opts = ferrostat_parser::parser::split_options(rest);
        s.sel = match m.as_str() {
            "cv" => Sel::Cv,
            "adaptive" | "adapt" => Sel::Adaptive,
            "plugin" | "plug" => Sel::Plugin,
            "bic" => Sel::Bic,
            "none" => Sel::None,
            o => return Err(rc_err(198, format!("selection({o}) not allowed"))),
        };
        if let Some(f) = sub_num(&opts, "folds", 4)? {
            s.folds = f as usize;
        }
        if let Some(f) = sub_num(&opts, "steps", 4)? {
            s.steps = f.max(1.0) as usize;
        }
        if let Some(f) = sub_num(&opts, "power", 3)? {
            s.power = f;
        }
        s.serule = sub_opt(&opts, "serule", 3).is_some();
        s.homoskedastic = sub_opt(&opts, "homoskedastic", 4).is_some();
    }
    if let Some(f) = util::opt_num(cmd, "folds", 4)? {
        s.folds = f as usize;
    }
    if s.folds < 2 {
        return Err(rc_err(198, "folds() must be at least 2"));
    }
    if let Some(g) = cmd.opt("grid", 4) {
        let (a, rest) = match g.find(',') {
            Some(p) => (g[..p].trim(), &g[p + 1..]),
            None => (g.trim(), ""),
        };
        s.grid_n = a.parse::<f64>().map_err(|_| rc_err(198, "invalid grid()"))?.max(2.0) as usize;
        let opts = ferrostat_parser::parser::split_options(rest);
        s.ratio = sub_num(&opts, "ratio", 3)?;
        s.lmin = sub_num(&opts, "min", 3)?;
    }
    if let Some(v) = util::opt_num(cmd, "stop", 4)? {
        s.stop = v;
    }
    if let Some(l) = cmd.opt("lambda", 6) {
        let mut v: Vec<f64> = l.split_whitespace().map(|x| x.parse::<f64>().map_err(|_| rc_err(198, "invalid lambda()"))).collect::<Result<_, _>>()?;
        v.sort_by(|a, b| b.total_cmp(a));
        s.lambdas = Some(v);
    }
    Ok(s)
}

/// Reseed the session generator from rseed(#).
pub fn rseed(cmd: &Command, st: &Store) -> Result<(), String> {
    if let Some(v) = util::opt_num(cmd, "rseed", 5)? {
        st.rng.borrow_mut().reseed(v as u64);
    }
    Ok(())
}

pub fn sel_label(l: &Lasso) -> String {
    match l.selection.as_str() {
        "cv" => "Cross-validation".into(),
        "adaptive" => "Adaptive".into(),
        "plugin" => format!("Plugin {}", if l.how == "homoskedastic" { "homoskedastic" } else { "heteroskedastic" }),
        "bic" => "Bayesian information criterion".into(),
        _ => "None".into(),
    }
}

// ---------------------------------------------------------------------------
// running a selection
// ---------------------------------------------------------------------------

fn to_pt(alpha: f64, p: &Point) -> Pt {
    Pt { alpha, lambda: p.lambda, b0: p.b0, b: p.b.clone(), dev: p.dev, df: p.df, cvm: f64::NAN, cvse: f64::NAN, bic: f64::NAN }
}

/// Square-root lasso path with warm scale.
fn sqrt_path(p: &Problem, lambdas: &[f64]) -> Vec<Point> {
    let mut sigma = sd_y(p);
    let mut out = Vec::new();
    let mut w = core::Warm::new(p);
    for &l in lambdas {
        let (pt, s) = core::sqrt_fit_warm(&mut w, l, sigma);
        sigma = s;
        out.push(pt);
    }
    out
}

fn sd_y(p: &Problem) -> f64 {
    let sw: f64 = p.w.iter().sum::<f64>().max(1e-300);
    let m = p.y.iter().zip(&p.w).map(|(a, b)| a * b).sum::<f64>() / sw;
    (p.y.iter().zip(&p.w).map(|(a, b)| b * (a - m) * (a - m)).sum::<f64>() / sw).sqrt().max(1e-10)
}

fn null_dev(p: &Problem) -> f64 {
    let q = Problem { loadings: p.loadings.iter().zip(&p.loadings).map(|(l, _)| if *l == 0.0 { 0.0 } else { 1.0 }).collect(), ..p.clone() };
    let lmax = core::lambda_max(&q);
    core::path(&q, &[lmax * 1.0001 + 1e-12], 0.0).pop().map_or(f64::NAN, |pt| pt.dev)
}

/// Fit `p` (loadings set: 0 for always-included covariates) with the
/// selection method in `s`; `uniform` supplies draws for CV folds.
pub fn run(kind: &str, depvar: &str, names: &[String], always: &[bool], p: &Problem, s: &Settings, uniform: &mut dyn FnMut() -> f64) -> Result<Lasso, String> {
    let n = p.y.len();
    let np = p.x.len();
    let mut l = Lasso {
        family: p.family,
        kind: kind.into(),
        depvar: depvar.into(),
        names: names.to_vec(),
        always: always.to_vec(),
        loadings: p.loadings.clone(),
        selection: match s.sel {
            Sel::Cv => "cv",
            Sel::Adaptive => "adaptive",
            Sel::Plugin => "plugin",
            Sel::Bic => "bic",
            Sel::None => "none",
        }
        .into(),
        n,
        nfolds: if matches!(s.sel, Sel::Cv | Sel::Adaptive) { s.folds } else { 0 },
        sel: None,
        dev0: null_dev(p),
        pts: Vec::new(),
        how: String::new(),
    };
    if s.sel == Sel::Plugin {
        let (lam, load, pt) = if s.sqrt {
            sqrt_plugin(p, s.homoskedastic)
        } else {
            core::plugin(p, s.homoskedastic)
        };
        l.loadings = load;
        let mut q = to_pt(1.0, &pt);
        q.lambda = lam;
        l.pts.push(q);
        l.sel = Some(0);
        l.how = if s.homoskedastic { "homoskedastic".into() } else { "heteroskedastic".into() };
        return Ok(l);
    }
    let fold = if matches!(s.sel, Sel::Cv | Sel::Adaptive) {
        if s.folds > n {
            return Err(rc_err(198, "folds() larger than the number of observations"));
        }
        core::folds(n, s.folds, uniform)
    } else {
        Vec::new()
    };
    let mut prob = p.clone();
    let steps = if s.sel == Sel::Adaptive { s.steps } else { 1 };
    for step in 0..steps {
        let mut pts: Vec<Pt> = Vec::new();
        for &alpha in &s.alphas {
            prob.alpha = alpha;
            let lams: Vec<f64> = match &s.lambdas {
                Some(v) => v.clone(),
                None => {
                    let mut lmax = core::lambda_max(&prob);
                    if s.sqrt {
                        lmax /= sd_y(&prob);
                    }
                    if !(lmax > 0.0) {
                        lmax = 1.0;
                    }
                    let ratio = match (s.lmin, s.ratio) {
                        (Some(m), _) => (m / lmax).min(1.0),
                        (None, Some(r)) => r,
                        (None, None) => core::default_ratio(n, np),
                    };
                    core::grid(lmax, s.grid_n, ratio)
                }
            };
            let path: Vec<Point> = if s.sqrt { sqrt_path(&prob, &lams) } else { core::path(&prob, &lams, if s.lambdas.is_some() { 0.0 } else { s.stop }) };
            let used: Vec<f64> = path.iter().map(|q| q.lambda).collect();
            let mut part: Vec<Pt> = path.iter().map(|q| to_pt(alpha, q)).collect();
            if !fold.is_empty() {
                let (m, se) = if s.sqrt { core::cv_with(&prob, &used, &fold, s.folds, &|q, ls| sqrt_path(q, ls)) } else { core::cv(&prob, &used, &fold, s.folds) };
                for (k, q) in part.iter_mut().enumerate() {
                    q.cvm = m[k];
                    q.cvse = se[k];
                }
            }
            for (q, pp) in part.iter_mut().zip(&path) {
                q.bic = core::bic(&prob, pp);
            }
            pts.extend(part);
        }
        // choose
        let sel = match s.sel {
            Sel::Cv | Sel::Adaptive => {
                let best = (0..pts.len()).filter(|&k| pts[k].cvm.is_finite()).min_by(|&a, &b| pts[a].cvm.total_cmp(&pts[b].cvm));
                match (best, s.serule) {
                    (Some(b), true) => {
                        let thr = pts[b].cvm + pts[b].cvse;
                        let a = pts[b].alpha;
                        (0..pts.len()).filter(|&k| pts[k].alpha == a && pts[k].cvm <= thr && pts[k].lambda >= pts[b].lambda).max_by(|&x, &y| pts[x].lambda.total_cmp(&pts[y].lambda)).or(Some(b))
                    }
                    (b, _) => b,
                }
            }
            Sel::Bic => (0..pts.len()).filter(|&k| pts[k].bic.is_finite()).min_by(|&a, &b| pts[a].bic.total_cmp(&pts[b].bic)),
            _ => None,
        };
        l.pts = pts;
        l.sel = sel;
        l.how = match s.sel {
            Sel::Cv | Sel::Adaptive if s.serule => "serule".into(),
            Sel::Cv | Sel::Adaptive => "cv".into(),
            Sel::Bic => "bic".into(),
            _ => String::new(),
        };
        l.loadings = prob.loadings.clone();
        if step + 1 < steps {
            // adaptive weights from the selected standardized coefficients
            let Some(k) = sel else { break };
            let pt = &l.pts[k];
            if pt.df == 0 {
                break;
            }
            let sds: Vec<f64> = p.x.iter().map(|c| sd_col(c, &p.w)).collect();
            prob.loadings = (0..np)
                .map(|j| {
                    if p.loadings[j] == 0.0 {
                        0.0
                    } else if pt.b[j] == 0.0 {
                        1e12
                    } else {
                        p.loadings[j] / (pt.b[j] * sds[j]).abs().powf(s.power)
                    }
                })
                .collect();
        }
    }
    Ok(l)
}

fn sd_col(c: &[f64], w: &[f64]) -> f64 {
    let sw: f64 = w.iter().sum::<f64>().max(1e-300);
    let m = c.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() / sw;
    (c.iter().zip(w).map(|(a, b)| b * (a - m) * (a - m)).sum::<f64>() / sw).sqrt()
}

/// Plug-in square-root lasso: pivotal lambda (no scale estimate needed);
/// loadings from residuals scaled by their root mean square.
fn sqrt_plugin(p: &Problem, homoskedastic: bool) -> (f64, Vec<f64>, Point) {
    let n = p.y.len() as f64;
    let np = p.x.len().max(1) as f64;
    let gamma = 0.1 / np.max(n).ln();
    let lam = 1.1 * crate::dist::invnormal(1.0 - gamma / (2.0 * np)) / n.sqrt();
    let sds: Vec<f64> = p.x.iter().map(|c| sd_col(c, &p.w)).collect();
    let sw: f64 = p.w.iter().sum::<f64>().max(1e-300);
    let ybar = p.y.iter().zip(&p.w).map(|(a, b)| a * b).sum::<f64>() / sw;
    let mut resid: Vec<f64> = p.y.iter().map(|y| y - ybar).collect();
    let mut q = p.clone();
    q.standardize = true;
    let mut load = vec![1.0; p.x.len()];
    let mut pt = core::sqrt_fit(&q, lam, sd_y(p)).0;
    for _ in 0..15 {
        let s2 = resid.iter().zip(&p.w).map(|(r, w)| w * r * r).sum::<f64>() / sw;
        let new: Vec<f64> = (0..p.x.len())
            .map(|j| {
                if p.loadings[j] == 0.0 || sds[j] == 0.0 {
                    return 0.0;
                }
                if homoskedastic {
                    return p.loadings[j];
                }
                let m = p.x[j].iter().zip(&p.w).map(|(a, b)| a * b).sum::<f64>() / sw;
                let v = (0..p.y.len()).map(|i| p.w[i] * ((p.x[j][i] - m) / sds[j]).powi(2) * resid[i] * resid[i]).sum::<f64>() / sw;
                p.loadings[j] * (v / s2.max(1e-300)).sqrt()
            })
            .collect();
        let ch = new.iter().zip(&load).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        load = new;
        q.loadings = load.clone();
        pt = core::sqrt_fit(&q, lam, s2.sqrt()).0;
        let sel: Vec<usize> = (0..p.x.len()).filter(|&j| pt.b[j] != 0.0).collect();
        let (c0, cb) = core::refit(p, &sel);
        resid = (0..p.y.len()).map(|i| p.y[i] - c0 - sel.iter().map(|&j| cb[j] * p.x[j][i]).sum::<f64>()).collect();
        if ch < 1e-6 {
            break;
        }
    }
    (lam, load, pt)
}

/// Problem from built data with base loadings (0 for always-included).
pub fn problem(fam: Family, d: &Data, y: Vec<f64>, alpha: f64, standardize: bool) -> Problem {
    Problem { family: fam, x: d.x.clone(), y, w: d.w.clone(), offset: None, loadings: d.always.iter().map(|a| if *a { 0.0 } else { 1.0 }).collect(), alpha, standardize }
}

/// Post-selection (unpenalized) coefficients at point `k` over `p`.
pub fn postsel(l: &Lasso, k: usize, p: &Problem) -> (f64, Vec<f64>) {
    let sel = l.nonzero(k);
    let mut sel2 = sel.clone();
    for (j, a) in l.always.iter().enumerate() {
        if *a && !sel2.contains(&j) {
            sel2.push(j);
        }
    }
    sel2.sort_unstable();
    core::refit(p, &sel2)
}

/// Rebuild the covariate columns and outcome of a stored lasso from the
/// data, over observations flagged in `mask` (missing covariates: None).
pub fn columns(ds: &DataSet, st: &Store, l: &Lasso, mask: &[bool]) -> Result<(Vec<usize>, Vec<Vec<f64>>, Vec<f64>), String> {
    let env = crate::eval::Env::with_store(ds, st);
    let y = util::col(ds, &l.depvar).unwrap_or_else(|_| vec![SYSMISS; ds.n_obs]);
    let mut rows = Vec::new();
    let mut x: Vec<Vec<f64>> = vec![Vec::new(); l.names.len()];
    let mut yy = Vec::new();
    for i in 0..ds.n_obs {
        if !mask[i] {
            continue;
        }
        let v: Vec<f64> = l.names.iter().map(|n| crate::postest::coef_value(&env, n, i)).collect::<Result<_, _>>()?;
        if v.iter().any(|a| is_missing(*a)) {
            continue;
        }
        rows.push(i);
        for (j, a) in v.into_iter().enumerate() {
            x[j].push(a);
        }
        yy.push(y[i]);
    }
    Ok((rows, x, yy))
}
