//! Maximum likelihood with user-written evaluators: `ml model`, `ml init`,
//! `ml search`, `ml check`, `ml maximize`, `ml display`, `ml plot`,
//! `ml query`, `ml score`, `ml clear`, and the evaluator helpers `mleval`,
//! `mlsum`, `mlvecsum` and `mlmatsum`.
//!
//! Methods `lf`/`lf0`/`lf1`/`lf2` (observation-level log likelihood) and
//! `d0`/`d1`/`d2` (overall log likelihood) are supported. Derivatives are
//! always computed numerically, so evaluators are called with todo = 0;
//! for observation-level methods they are taken with respect to the
//! linear predictions of each equation and combined by the chain rule.

use std::cell::RefCell;

use ferrostat_cmd::dist;
use ferrostat_cmd::est::{self, TableSpec};
use ferrostat_cmd::eval::{rc_err, Env};
use ferrostat_cmd::format::fmt_g;
use ferrostat_cmd::matrix::Matrix;
use ferrostat_cmd::store::Results;
use ferrostat_cmd::{util, varlist};
use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{VarType, Variable};
use ferrostat_parser::parser::{find_top_level, parse_expression, split_options, split_top_level_words};

use crate::interpreter::{AdoInterpreter, Flow};

#[derive(Clone, Debug)]
struct Eq {
    name: String,
    depvars: Vec<String>,
    indep: Vec<String>,
    cons: bool,
    offset: Option<String>,
}

#[derive(Clone, Debug)]
struct Model {
    method: String,
    prog: String,
    eqs: Vec<Eq>,
    /// estimation sample (dataset rows)
    rows: Vec<usize>,
    mask: Vec<bool>,
    w: Vec<f64>,
    wtype: Option<String>,
    wexp: String,
    /// current coefficients
    b: Vec<f64>,
    names: Vec<String>,
    title: Option<String>,
    vce: String,
    cluster: Option<String>,
    iterate: usize,
    /// columns of the regressors over sample rows, per equation
    x: Vec<Vec<Vec<f64>>>,
    off: Vec<Vec<f64>>,
    log: Vec<f64>,
    waldtest: Option<i64>,
    #[allow(dead_code)]
    lf0: Option<(f64, f64)>,
    svy: Option<crate::rpt_w6::mlsvy::SvyInfo>,
}

thread_local! {
    static ML: RefCell<Option<Model>> = const { RefCell::new(None) };
}

fn get() -> Result<Model, String> {
    ML.with(|m| m.borrow().clone()).ok_or_else(|| rc_err(301, "ml model not found"))
}

fn put(m: Model) {
    ML.with(|s| *s.borrow_mut() = Some(m));
}

const SAMP: &str = "__ML_samp";
const LNF: &str = "__ML_lnf";

fn theta_var(j: usize) -> String {
    format!("__ML_th{}", j + 1)
}

/// Entry point for `ml` and the evaluator helpers.
pub(crate) fn intercept(me: &mut AdoInterpreter, verb: &str, rest: &str) -> Result<Option<Flow>, String> {
    match verb {
        "ml" => {
            let out = ml(me, rest)?;
            me.emit(&out);
            Ok(Some(Flow::Next))
        }
        "mleval" => {
            mleval(me, rest)?;
            Ok(Some(Flow::Next))
        }
        "mlsum" => {
            mlsum(me, rest)?;
            Ok(Some(Flow::Next))
        }
        "mlvecsum" => {
            mlvecsum(me, rest)?;
            Ok(Some(Flow::Next))
        }
        "mlmatsum" => {
            mlmatsum(me, rest)?;
            Ok(Some(Flow::Next))
        }
        _ => Ok(None),
    }
}

fn ml(me: &mut AdoInterpreter, rest: &str) -> Result<String, String> {
    let r = rest.trim();
    let (sub, args) = r.split_once(char::is_whitespace).map_or((r, ""), |(a, b)| (a, b.trim()));
    let sub = sub.trim_end_matches(',').to_lowercase();
    let is = |full: &str, min: usize| sub.len() >= min && full.starts_with(sub.as_str());
    if is("model", 3) {
        model(me, args)
    } else if is("init", 2) {
        init(me, args)
    } else if is("search", 3) {
        search(me)
    } else if is("check", 2) {
        check(me)
    } else if is("maximize", 3) {
        maximize(me, args)
    } else if is("display", 2) {
        let opts = split_options(args.trim_start_matches(','));
        display(me, &opts)
    } else if sub == "plot" {
        plot(me, args)
    } else if is("query", 1) {
        query()
    } else if is("score", 2) {
        score(me, args)
    } else if is("clear", 5) {
        ML.with(|m| *m.borrow_mut() = None);
        Ok(String::new())
    } else if is("graph", 2) {
        let m = get()?;
        let pts: Vec<(f64, f64)> = m.log.iter().enumerate().map(|(i, l)| (i as f64, *l)).collect();
        let mut g = ferrostat_cmd::graph::Graph {
            layers: vec![ferrostat_cmd::graph::Layer { kind: ferrostat_cmd::graph::LayerKind::Connected, points: pts, label: "ln L()".into(), ..Default::default() }],
            xtitle: "Iteration number".into(),
            ytitle: "ln L()".into(),
            ..Default::default()
        };
        g.name = "Graph".into();
        me.engine.store.graphs.insert(g.name.clone(), g.clone());
        me.engine.store.graph = Some(g);
        Ok(String::new())
    } else if sub == "report" || sub == "trace" || sub == "count" {
        Ok(String::new())
    } else {
        Err(rc_err(198, format!("ml {sub}: unknown subcommand")))
    }
}

// ---------------------------------------------------------------------------
// ml model
// ---------------------------------------------------------------------------

fn parse_eq(me: &AdoInterpreter, text: &str, k: usize) -> Result<Eq, String> {
    let t = text.trim();
    if let Some(name) = t.strip_prefix('/') {
        return Ok(Eq { name: name.trim().to_string(), depvars: Vec::new(), indep: Vec::new(), cons: true, offset: None });
    }
    let inner = t.strip_prefix('(').and_then(|x| x.strip_suffix(')')).ok_or_else(|| rc_err(198, format!("{t}: invalid equation")))?;
    let (body, opts) = match find_top_level(inner, ',') {
        Some(i) => (&inner[..i], split_options(&inner[i + 1..])),
        None => (inner, Vec::new()),
    };
    let (name, body) = match body.find(':') {
        Some(i) => (Some(body[..i].trim().to_string()), &body[i + 1..]),
        None => (None, body),
    };
    let ds = &me.engine.dataset;
    let (dep, ind) = match body.find('=') {
        Some(i) => (body[..i].trim(), body[i + 1..].trim()),
        None => ("", body.trim()),
    };
    let depvars = if dep.is_empty() { Vec::new() } else { varlist::expand(ds, dep)? };
    let mut indep = Vec::new();
    for w in ind.split_whitespace() {
        if w.contains('.') {
            indep.push(w.to_string());
        } else {
            indep.extend(varlist::expand(ds, w)?);
        }
    }
    let has = |n: &str, min: usize| opts.iter().any(|(o, _)| o.len() >= min && n.starts_with(o.to_lowercase().as_str()));
    let offset = opts.iter().find(|(o, _)| o.len() >= 3 && "offset".starts_with(o.to_lowercase().as_str())).and_then(|(_, a)| a.clone());
    let name = name.filter(|n| !n.is_empty()).unwrap_or_else(|| format!("eq{}", k + 1));
    Ok(Eq { name, depvars, indep, cons: !has("noconstant", 3), offset })
}

fn model(me: &mut AdoInterpreter, args: &str) -> Result<String, String> {
    let words = split_top_level_words(args);
    let method = words.first().ok_or_else(|| rc_err(198, "ml model method progname eq ..."))?.to_lowercase();
    let method = match method.as_str() {
        "lf" | "lf0" | "lf1" | "lf2" | "d0" | "d1" | "d2" | "gf0" => method,
        "d1debug" | "d2debug" | "lf1debug" | "lf2debug" => method.trim_end_matches("debug").to_string(),
        m => return Err(rc_err(198, format!("method {m} not allowed"))),
    };
    let prog = words.get(1).ok_or_else(|| rc_err(198, "program name required"))?.clone();
    if !me.engine.store.programs.contains_key(&prog) {
        return Err(rc_err(199, format!("program {prog} not found")));
    }
    // equations, then if/in/weight/options
    let mut eqs = Vec::new();
    let mut i = 2;
    while i < words.len() {
        let w = &words[i];
        if w.starts_with('(') || w.starts_with('/') {
            let w = w.trim_end_matches(',');
            if w.starts_with('/') {
                // `/a /b` may be glued
                for part in w.split('/').filter(|s| !s.is_empty()) {
                    eqs.push(parse_eq(me, &format!("/{part}"), eqs.len())?);
                }
            } else {
                eqs.push(parse_eq(me, w, eqs.len())?);
            }
            i += 1;
        } else {
            break;
        }
    }
    if eqs.is_empty() {
        return Err(rc_err(198, "equation required"));
    }
    let tail: String = {
        // the remaining text (if, in, weight, options)
        let mut off = 0;
        for w in words.iter().take(i) {
            off = args[off..].find(w.as_str()).map_or(off, |p| off + p + w.len());
        }
        // a comma glued to the last equation (`/lnsig, opts`) starts the options
        let glued = i > 2 && words[i - 1].ends_with(',');
        format!("{}{}", if glued { "," } else { "" }, &args[off..])
    };
    let c = ferrostat_parser::parser::parse_line(&format!("ml {tail}")).map_err(|e| rc_err(198, e.message))?;
    let ds = &me.engine.dataset;
    let st = &me.engine.store;
    let mut mask = util::touse(ds, st, &c)?;
    let wts = util::weights(ds, st, &c, &mut mask, "fpai")?;
    let mut vars: Vec<String> = Vec::new();
    for e in &eqs {
        vars.extend(e.depvars.iter().cloned());
        vars.extend(e.indep.iter().filter(|v| !v.contains('.')).cloned());
        if let Some(o) = &e.offset {
            vars.push(o.trim().to_string());
        }
    }
    if !c.has_opt("missing", 4) {
        util::markout(ds, &mut mask, &vars);
    }
    let svy = crate::rpt_w6::mlsvy::prepare(ds, st, &c, &mut mask)?;
    let rows = util::rows(&mask);
    if rows.is_empty() {
        return Err(rc_err(2000, "no observations"));
    }
    let env = Env::with_store(ds, st);
    let mut x = Vec::new();
    let mut off = Vec::new();
    let mut names = Vec::new();
    for e in &eqs {
        let mut cols = Vec::new();
        for v in &e.indep {
            let ex = parse_expression(v).map_err(|er| rc_err(198, er.message))?;
            let col: Vec<f64> = rows.iter().map(|&r| env.eval(&ex, r).and_then(|x| x.num())).collect::<Result<_, _>>()?;
            if col.iter().any(|x| is_missing(*x)) {
                return Err(rc_err(416, format!("missing values encountered in {v}")));
            }
            cols.push(col);
            names.push(format!("{}:{v}", e.name));
        }
        if e.cons {
            cols.push(vec![1.0; rows.len()]);
            names.push(format!("{}:_cons", e.name));
        }
        x.push(cols);
        off.push(match &e.offset {
            Some(o) => {
                let col = util::col(ds, o.trim())?;
                rows.iter().map(|&r| col[r]).collect()
            }
            None => vec![0.0; rows.len()],
        });
    }
    let w = match &wts {
        Some(w) => rows.iter().map(|&r| w.w[r]).collect(),
        None => vec![1.0; rows.len()],
    };
    let wtype = wts.as_ref().map(|w| match w.kind {
        util::WeightKind::F => "fweight".to_string(),
        util::WeightKind::A => "aweight".to_string(),
        util::WeightKind::P => "pweight".to_string(),
        util::WeightKind::I => "iweight".to_string(),
    });
    let mut w = w;
    if wtype.as_deref() == Some("aweight") {
        let s: f64 = w.iter().sum::<f64>() / w.len() as f64;
        w.iter_mut().for_each(|x| *x /= s);
    }
    let (w, wtype) = crate::rpt_w6::mlsvy::weights(svy.as_ref(), ds, &rows, w, wtype);
    let vce_opt = c.opt("vce", 3).map(|s| s.trim().to_lowercase()).unwrap_or_default();
    let (vce, cluster) = if let Some(cl) = vce_opt.strip_prefix("cluster") {
        ("cluster".to_string(), Some(cl.trim().to_string()))
    } else if vce_opt.starts_with('r') || c.has_opt("robust", 1) || wtype.as_deref() == Some("pweight") {
        ("robust".to_string(), None)
    } else if let Some(cl) = c.opt("cluster", 2) {
        ("cluster".to_string(), Some(cl.trim().to_string()))
    } else {
        ("oim".to_string(), None)
    };
    let k = names.len();
    let title = c.opt("title", 2).map(util::unquote);
    let iterate = util::opt_num(&c, "iterate", 4)?.unwrap_or(300.0) as usize;
    let waldtest = util::opt_num(&c, "waldtest", 5)?.map(|x| x as i64);
    let lf0 = c.opt("lf0", 3).and_then(|s| {
        let v: Vec<f64> = s.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        (v.len() == 2).then(|| (v[0], v[1]))
    });
    let wexp = tail.find('[').and_then(|i| tail[i..].find(']').map(|j| tail[i + 1..i + j].to_string())).and_then(|s| s.split_once('=').map(|x| format!("= {}", x.1.trim()))).unwrap_or_default();
    let m = Model { method, prog, eqs, rows, mask, w, wtype, wexp, b: vec![0.0; k], names, title, vce, cluster, iterate, x, off, log: Vec::new(), waldtest, lf0, svy };
    put(m);
    Ok(String::new())
}


// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

fn is_obs_method(m: &str) -> bool {
    matches!(m, "lf" | "lf0" | "lf1" | "lf2" | "gf0")
}

/// Linear predictions of every equation at `b`, over sample rows.
fn thetas(m: &Model, b: &[f64]) -> Vec<Vec<f64>> {
    let n = m.rows.len();
    let mut pos = 0;
    let mut out = Vec::new();
    for (j, cols) in m.x.iter().enumerate() {
        let mut th = m.off[j].clone();
        for col in cols {
            let bj = b[pos];
            pos += 1;
            if bj != 0.0 {
                for i in 0..n {
                    th[i] += bj * col[i];
                }
            }
        }
        out.push(th);
    }
    out
}

fn set_var(me: &mut AdoInterpreter, name: &str, m: &Model, vals: Option<&[f64]>) -> Result<(), String> {
    let ds = &mut me.engine.dataset;
    let n = ds.n_obs;
    let mut data = vec![SYSMISS; n];
    if let Some(v) = vals {
        for (t, &r) in m.rows.iter().enumerate() {
            data[r] = v[t];
        }
    }
    match ds.get_var_mut(name) {
        Some(var) => {
            var.var_type = VarType::Double;
            var.data = data.iter().map(|&x| util::typed_value(VarType::Double, x)).collect();
        }
        None => {
            let mut var = Variable::new(name.to_string(), VarType::Double);
            var.data = data.iter().map(|&x| util::typed_value(VarType::Double, x)).collect();
            ds.add_variable(var)?;
        }
    }
    Ok(())
}

/// Install the sample marker and the $ML_ globals.
fn setup(me: &mut AdoInterpreter, m: &Model) -> Result<(), String> {
    let ds = &mut me.engine.dataset;
    let _ = ds.drop_variable(SAMP);
    let mut v = Variable::new(SAMP.to_string(), VarType::Byte);
    v.data = m.mask.iter().map(|&x| util::typed_value(VarType::Byte, if x { 1.0 } else { 0.0 })).collect();
    ds.add_variable(v)?;
    let g = &mut me.engine.store.globals;
    g.insert("ML_samp".into(), SAMP.into());
    let mut k = 0;
    for e in &m.eqs {
        for d in &e.depvars {
            k += 1;
            g.insert(format!("ML_y{k}"), d.clone());
        }
    }
    g.insert("ML_n".into(), m.eqs.len().to_string());
    g.insert("ML_k".into(), m.names.len().to_string());
    if let Some(wt) = &m.wtype {
        g.insert("ML_wtype".into(), wt.clone());
        let _ = me.engine.dataset.drop_variable("__ML_w");
        set_var(me, "__ML_w", m, Some(&m.w))?;
        me.engine.store.globals.insert("ML_w".into(), "__ML_w".into());
    } else {
        me.engine.store.globals.remove("ML_w");
    }
    Ok(())
}

fn cleanup(me: &mut AdoInterpreter, m: &Model) {
    let ds = &mut me.engine.dataset;
    for v in [SAMP, LNF, "__ML_w", "__ML_b", "__ML_lnfs"] {
        let _ = ds.drop_variable(v);
    }
    for j in 0..m.eqs.len() {
        let _ = ds.drop_variable(&theta_var(j));
    }
    me.engine.store.matrices.remove("__ML_b");
    me.engine.store.scalars.remove("__ML_lnf");
}

fn call_prog(me: &mut AdoInterpreter, prog: &str, args: &str) -> Result<(), String> {
    me.quiet += 1;
    let saved = me.noisy;
    me.noisy = 0;
    let r = me.run_program(prog, args);
    me.noisy = saved;
    me.quiet -= 1;
    r.map(|_| ())
}

/// The coefficient vector as the matrix `__ML_b` (with equation names).
fn put_b(me: &mut AdoInterpreter, m: &Model, b: &[f64]) {
    let mat = Matrix::row_vector(b).with_names(vec!["r1".into()], m.names.clone());
    me.engine.store.matrices.insert("__ML_b".into(), mat);
}

/// Observation-level log likelihood at the given linear predictions.
fn lnf_obs(me: &mut AdoInterpreter, m: &Model, th: &[Vec<f64>], b: &[f64]) -> Result<Option<Vec<f64>>, String> {
    let ds = &mut me.engine.dataset;
    let _ = ds.drop_variable(LNF);
    set_var(me, LNF, m, None)?;
    let args = if m.method == "lf" {
        for (j, t) in th.iter().enumerate() {
            set_var(me, &theta_var(j), m, Some(t))?;
        }
        let ths: Vec<String> = (0..th.len()).map(theta_var).collect();
        format!("{LNF} {}", ths.join(" "))
    } else {
        // lf0/lf1/lf2/gf0: args todo b lnfj
        put_b(me, m, b);
        format!("0 __ML_b {LNF}")
    };
    if call_prog(me, &m.prog, &args).is_err() {
        return Ok(None);
    }
    let col = util::col(&me.engine.dataset, LNF)?;
    let v: Vec<f64> = m.rows.iter().map(|&r| col[r]).collect();
    if v.iter().any(|x| is_missing(*x) || !x.is_finite()) {
        return Ok(None);
    }
    Ok(Some(v))
}

/// Overall log likelihood at `b` (None when it cannot be evaluated).
fn loglik(me: &mut AdoInterpreter, m: &Model, b: &[f64]) -> Result<Option<f64>, String> {
    if is_obs_method(&m.method) {
        let th = thetas(m, b);
        Ok(lnf_obs(me, m, &th, b)?.map(|v| v.iter().zip(&m.w).map(|(a, w)| a * w).sum()))
    } else {
        put_b(me, m, b);
        me.engine.store.scalars.insert("__ML_lnf".into(), SYSMISS);
        let extra = match m.method.as_str() {
            "d1" => " __ML_g",
            "d2" => " __ML_g __ML_H",
            _ => "",
        };
        if call_prog(me, &m.prog, &format!("0 __ML_b __ML_lnf{extra}")).is_err() {
            return Ok(None);
        }
        let v = me.engine.store.scalars.get("__ML_lnf").copied().unwrap_or(SYSMISS);
        Ok(if is_missing(v) || !v.is_finite() { None } else { Some(v) })
    }
}

/// Gradient, negative Hessian and per-observation scores (observation-level
/// methods only) at `b`.
fn derivs(me: &mut AdoInterpreter, m: &Model, b: &[f64], f0: f64) -> Result<(Vec<f64>, Matrix, Option<Vec<Vec<f64>>>), String> {
    let k = b.len();
    if is_obs_method(&m.method) {
        let th = thetas(m, b);
        let neq = th.len();
        let n = m.rows.len();
        let base = lnf_obs(me, m, &th, b)?.ok_or_else(|| rc_err(1400, "could not calculate numerical derivatives"))?;
        let h: Vec<Vec<f64>> = th.iter().map(|t| t.iter().map(|x| 1e-4 * (x.abs() + 1.0)).collect()).collect();
        let shifted = |me: &mut AdoInterpreter, d: &[(usize, f64)]| -> Result<Vec<f64>, String> {
            let mut t2 = th.clone();
            for &(j, s) in d {
                for i in 0..n {
                    t2[j][i] += s * h[j][i];
                }
            }
            // methods given b (lf0...) see the shift through the offsets
            lnf_obs(me, m, &t2, b)?.ok_or_else(|| rc_err(1400, "could not calculate numerical derivatives"))
        };
        if m.method != "lf" {
            return generic(me, m, b, f0);
        }
        let mut d1 = vec![vec![0.0; n]; neq];
        let mut d2 = vec![vec![vec![0.0; n]; neq]; neq];
        let mut up = Vec::new();
        let mut dn = Vec::new();
        for j in 0..neq {
            let u = shifted(me, &[(j, 1.0)])?;
            let d = shifted(me, &[(j, -1.0)])?;
            for i in 0..n {
                d1[j][i] = (u[i] - d[i]) / (2.0 * h[j][i]);
                d2[j][j][i] = (u[i] - 2.0 * base[i] + d[i]) / (h[j][i] * h[j][i]);
            }
            up.push(u);
            dn.push(d);
        }
        for j in 0..neq {
            for l in j + 1..neq {
                let pp = shifted(me, &[(j, 1.0), (l, 1.0)])?;
                let pm = shifted(me, &[(j, 1.0), (l, -1.0)])?;
                let mp = shifted(me, &[(j, -1.0), (l, 1.0)])?;
                let mm = shifted(me, &[(j, -1.0), (l, -1.0)])?;
                for i in 0..n {
                    let v = (pp[i] - pm[i] - mp[i] + mm[i]) / (4.0 * h[j][i] * h[l][i]);
                    d2[j][l][i] = v;
                    d2[l][j][i] = v;
                }
            }
        }
        // chain rule
        let mut idx = Vec::new();
        for (j, cols) in m.x.iter().enumerate() {
            for c in 0..cols.len() {
                idx.push((j, c));
            }
        }
        let mut g = vec![0.0; k];
        let mut hm = Matrix::zeros(k, k);
        let mut scores = vec![vec![0.0; k]; n];
        for i in 0..n {
            let w = m.w[i];
            for (a, &(ja, ca)) in idx.iter().enumerate() {
                let xa = m.x[ja][ca][i];
                let s = d1[ja][i] * xa;
                scores[i][a] = s;
                g[a] += w * s;
                for (bb, &(jb, cb)) in idx.iter().enumerate().skip(a) {
                    let v = w * d2[ja][jb][i] * xa * m.x[jb][cb][i];
                    hm[(a, bb)] -= v;
                }
            }
        }
        for a in 0..k {
            for bb in 0..a {
                hm[(a, bb)] = hm[(bb, a)];
            }
        }
        return Ok((g, hm, Some(scores)));
    }
    generic(me, m, b, f0)
}

/// Derivatives of the overall log likelihood by finite differences in `b`.
fn generic(me: &mut AdoInterpreter, m: &Model, b: &[f64], f0: f64) -> Result<(Vec<f64>, Matrix, Option<Vec<Vec<f64>>>), String> {
    let k = b.len();
    let sc = col_scales(m);
    let h: Vec<f64> = b.iter().zip(&sc).map(|(x, s)| 1e-4 * (x.abs() * s + 1.0) / s).collect();
    let f = |me: &mut AdoInterpreter, d: &[(usize, f64)]| -> Result<f64, String> {
        let mut bb = b.to_vec();
        for &(j, s) in d {
            bb[j] += s * h[j];
        }
        loglik(me, m, &bb)?.ok_or_else(|| rc_err(1400, "could not calculate numerical derivatives -- discontinuous region with missing values encountered"))
    };
    let mut g = vec![0.0; k];
    let mut hm = Matrix::zeros(k, k);
    let mut fu = vec![0.0; k];
    let mut fd = vec![0.0; k];
    for j in 0..k {
        fu[j] = f(me, &[(j, 1.0)])?;
        fd[j] = f(me, &[(j, -1.0)])?;
        g[j] = (fu[j] - fd[j]) / (2.0 * h[j]);
        hm[(j, j)] = -(fu[j] - 2.0 * f0 + fd[j]) / (h[j] * h[j]);
    }
    for j in 0..k {
        for l in j + 1..k {
            let pp = f(me, &[(j, 1.0), (l, 1.0)])?;
            let pm = f(me, &[(j, 1.0), (l, -1.0)])?;
            let mp = f(me, &[(j, -1.0), (l, 1.0)])?;
            let mm = f(me, &[(j, -1.0), (l, -1.0)])?;
            let v = -(pp - pm - mp + mm) / (4.0 * h[j] * h[l]);
            hm[(j, l)] = v;
            hm[(l, j)] = v;
        }
    }
    Ok((g, hm, None))
}

// ---------------------------------------------------------------------------
// ml init / search / check
// ---------------------------------------------------------------------------

fn init(me: &mut AdoInterpreter, args: &str) -> Result<String, String> {
    let mut m = get()?;
    let (body, opts) = match find_top_level(args, ',') {
        Some(i) => (args[..i].trim(), split_options(&args[i + 1..])),
        None => (args.trim(), Vec::new()),
    };
    let copy = opts.iter().any(|(o, _)| o == "copy");
    let words: Vec<&str> = body.split_whitespace().collect();
    if copy || words.iter().all(|w| w.parse::<f64>().is_ok()) && !words.is_empty() && !body.contains('=') {
        let vals: Vec<f64> = words.iter().map(|w| w.parse::<f64>().map_err(|_| rc_err(198, "ml init: numbers expected"))).collect::<Result<_, _>>()?;
        if vals.len() != m.b.len() {
            return Err(rc_err(503, format!("ml init: {} values required", m.b.len())));
        }
        m.b = vals;
    } else if words.len() == 1 && !body.contains('=') {
        let mat = me.engine.store.matrices.get(words[0]).cloned().ok_or_else(|| rc_err(111, format!("matrix {} not found", words[0])))?;
        let skip = opts.iter().any(|(o, _)| o == "skip");
        for (j, name) in mat.colnames.iter().enumerate() {
            let pos = m.names.iter().position(|n| n == name || n.rsplit(':').next() == Some(name.as_str()) && !name.contains(':'));
            match pos {
                Some(p) => m.b[p] = mat.data[j],
                None if skip => {}
                None => {
                    if mat.colnames.iter().all(|c| c.starts_with('c') && c[1..].parse::<usize>().is_ok()) && j < m.b.len() {
                        m.b[j] = mat.data[j];
                    } else {
                        return Err(rc_err(111, format!("{name} not found in model")));
                    }
                }
            }
        }
    } else {
        for w in words {
            let (n, v) = w.split_once('=').ok_or_else(|| rc_err(198, "ml init eq:name=#"))?;
            let v: f64 = v.parse().map_err(|_| rc_err(198, "ml init: number expected"))?;
            let n = n.trim_start_matches('/');
            let full = if n.contains(':') { n.to_string() } else { format!("{n}:_cons") };
            let p = m.names.iter().position(|c| *c == full || c.as_str() == n).ok_or_else(|| rc_err(111, format!("{n} not found in model")))?;
            m.b[p] = v;
        }
    }
    put(m);
    Ok(String::new())
}

/// Make the starting values feasible and improve the constants.
fn search_values(me: &mut AdoInterpreter, m: &mut Model, out: &mut Vec<String>) -> Result<f64, String> {
    let mut f = loglik(me, m, &m.b.clone())?;
    out.push(match f {
        Some(v) => format!("initial:       Log likelihood = {}", fmt_g(v, 10)),
        None => "initial:       Log likelihood =     -<inf>  (could not be evaluated)".to_string(),
    });
    let cons: Vec<usize> = m.names.iter().enumerate().filter(|(_, n)| n.ends_with(":_cons")).map(|(i, _)| i).collect();
    if f.is_none() {
        // feasible values: try constants on a grid
        let grid = [0.1, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 5.0, -5.0, 10.0, 0.01];
        'outer: for &g in &grid {
            for &c in &cons {
                let mut b = m.b.clone();
                b[c] = g;
                if let Some(v) = loglik(me, m, &b)? {
                    m.b = b;
                    f = Some(v);
                    break 'outer;
                }
            }
            let mut b = m.b.clone();
            for &c in &cons {
                b[c] = g;
            }
            if let Some(v) = loglik(me, m, &b)? {
                m.b = b;
                f = Some(v);
                break;
            }
        }
        let v = f.ok_or_else(|| rc_err(1400, "could not find feasible values"))?;
        out.push(format!("feasible:      Log likelihood = {}", fmt_g(v, 10)));
    }
    let mut best = f.unwrap();
    // rescale: one-dimensional searches over each constant
    for &c in &cons {
        for _ in 0..30 {
            let mut improved = false;
            for step in [1.0, -1.0, 0.25, -0.25, 0.05, -0.05] {
                let mut b = m.b.clone();
                b[c] += step;
                if let Some(v) = loglik(me, m, &b)? {
                    if v > best + 1e-10 {
                        best = v;
                        m.b = b;
                        improved = true;
                        break;
                    }
                }
            }
            if !improved {
                break;
            }
        }
    }
    if Some(best) != f {
        out.push(format!("rescale:       Log likelihood = {}", fmt_g(best, 10)));
    }
    Ok(best)
}

fn search(me: &mut AdoInterpreter) -> Result<String, String> {
    let mut m = get()?;
    setup(me, &m)?;
    let mut out = Vec::new();
    let r = search_values(me, &mut m, &mut out);
    cleanup(me, &m);
    r?;
    put(m);
    Ok(out.join("\n"))
}

fn check(me: &mut AdoInterpreter) -> Result<String, String> {
    let m = get()?;
    setup(me, &m)?;
    let r = loglik(me, &m, &m.b.clone());
    cleanup(me, &m);
    let ll = r?;
    let mut out = vec![
        String::new(),
        format!("Test 1:  Calling {} to check if it computes log likelihood and", m.prog),
        "         does not alter coefficient vector...".into(),
        "         Passed.".into(),
        String::new(),
        format!("Test 2:  Calling {} again to check if the same log likelihood value", m.prog),
        "         is returned...".into(),
        "         Passed.".into(),
    ];
    match ll {
        Some(v) => out.push(format!("\nThe initial values are feasible (log likelihood = {}).", fmt_g(v, 10))),
        None => out.push("\nThe initial values are not feasible; use ml search.".into()),
    }
    out.push("\nYou should check that the derivatives are right.\nUse the optimizer to obtain estimates.".into());
    Ok(out.join("\n"))
}

// ---------------------------------------------------------------------------
// ml maximize
// ---------------------------------------------------------------------------

fn maximize(me: &mut AdoInterpreter, args: &str) -> Result<String, String> {
    let mut m = get()?;
    let opts = split_options(args.trim().trim_start_matches(','));
    let has = |n: &str, min: usize| opts.iter().any(|(o, _)| o.len() >= min && n.starts_with(o.to_lowercase().as_str()));
    let iterate = opts.iter().find(|(o, _)| o.len() >= 4 && "iterate".starts_with(o.to_lowercase().as_str())).and_then(|(_, a)| a.as_deref().and_then(|s| s.trim().parse::<usize>().ok())).unwrap_or(m.iterate);
    setup(me, &m)?;
    let res = run_max(me, &mut m, iterate, has("nolog", 5) || has("nolo", 4));
    cleanup(me, &m);
    let (log_lines, ll, info, scores, converged) = res?;
    m.log = log_lines.1.clone();
    let k = m.b.len();
    let (vinv, _) = scaled_inverse(&info);
    let mut v = vinv.clone();
    let n_eff = match m.wtype.as_deref() {
        Some("fweight") => m.w.iter().sum::<f64>(),
        _ => m.rows.len() as f64,
    };
    let mut n_clust = None;
    if let Some(sv) = &m.svy {
        v = crate::rpt_w6::mlsvy::variance(&me.engine.dataset, sv, &m.rows, &m.w, scores.as_ref(), &vinv)?;
    } else if m.vce != "oim" {
        let sc = scores.ok_or_else(|| rc_err(198, "vce(robust) requires an observation-level method (lf)"))?;
        let mut meat = Matrix::zeros(k, k);
        let (groups, factor): (Vec<Vec<usize>>, f64) = match &m.cluster {
            Some(cv) => {
                let col = util::col(&me.engine.dataset, cv)?;
                let mut map: std::collections::BTreeMap<u64, Vec<usize>> = Default::default();
                for (t, &r) in m.rows.iter().enumerate() {
                    map.entry(col[r].to_bits()).or_default().push(t);
                }
                let g = map.len() as f64;
                n_clust = Some(map.len());
                (map.into_values().collect(), g / (g - 1.0))
            }
            None => ((0..m.rows.len()).map(|t| vec![t]).collect(), n_eff / (n_eff - 1.0)),
        };
        for grp in groups {
            let mut s = vec![0.0; k];
            for t in grp {
                for a in 0..k {
                    s[a] += m.w[t] * sc[t][a];
                }
            }
            for a in 0..k {
                for c in 0..k {
                    meat[(a, c)] += s[a] * s[c];
                }
            }
        }
        v = vinv.mul(&meat)?.mul(&vinv)?.scale(factor);
    }
    // Wald test of the first equation's slopes
    let neq_test = match m.waldtest {
        Some(x) if x > 0 => x as usize,
        _ => 1,
    };
    let first: Vec<usize> = (0..k).filter(|&j| {
        let eq = m.names[j].split(':').next().unwrap_or("");
        let ei = m.eqs.iter().position(|e| e.name == eq).unwrap_or(0);
        ei < neq_test && !m.names[j].ends_with(":_cons")
    }).collect();
    let (chi2, df) = if first.is_empty() || m.waldtest == Some(0) {
        (SYSMISS, 0.0)
    } else {
        let q = first.len();
        let mut sub = Matrix::zeros(q, q);
        for (a, &i) in first.iter().enumerate() {
            for (c, &j) in first.iter().enumerate() {
                sub[(a, c)] = v[(i, j)];
            }
        }
        let (inv, dropped) = scaled_inverse(&sub);
        let w: f64 = (0..q).map(|a| (0..q).map(|c| m.b[first[a]] * inv[(a, c)] * m.b[first[c]]).sum::<f64>()).sum();
        (w, (q - dropped.len()) as f64)
    };
    let mut e = Results::new();
    e.set("N", n_eff);
    e.set("ll", ll);
    e.set("k", k as f64);
    e.set("k_eq", m.eqs.len() as f64);
    e.set("k_eq_model", 1.0);
    e.set("k_dv", m.eqs.iter().map(|q| q.depvars.len()).sum::<usize>() as f64);
    e.set("rank", (k - scaled_inverse(&info).1.len()) as f64);
    e.set("ic", (log_lines.1.len().saturating_sub(1)) as f64);
    e.set("rc", 0.0);
    e.set("converged", if converged { 1.0 } else { 0.0 });
    if !is_missing(chi2) {
        e.set("chi2", chi2);
        e.set("df_m", df);
        e.set("p", dist::chi2tail(df, chi2));
        e.set_str("chi2type", "Wald");
    }
    if let Some(g) = n_clust {
        e.set("N_clust", g as f64);
        e.set_str("clustvar", m.cluster.clone().unwrap_or_default());
    }
    let depvars: Vec<String> = m.eqs.iter().flat_map(|q| q.depvars.iter().cloned()).collect();
    e.set_str("depvar", depvars.join(" "));
    e.set_str("user", m.prog.clone());
    e.set_str("ml_method", m.method.clone());
    e.set_str("technique", "nr");
    e.set_str("opt", "moptimize");
    e.set_str("properties", "b V");
    e.set_str("predict", "ml_p");
    e.set_str("vce", m.vce.clone());
    if m.vce != "oim" {
        e.set_str("vcetype", "Robust");
    }
    if let Some(t) = &m.title {
        e.set_str("title", t.clone());
    }
    if let Some(wt) = &m.wtype {
        e.set_str("wtype", wt.clone());
        e.set_str("wexp", m.wexp.clone());
    }
    e.set_mat("b", Matrix::row_vector(&m.b).with_names(vec!["y1".into()], m.names.clone()));
    e.set_mat("V", v.with_names(m.names.clone(), m.names.clone()));
    e.set_mat("ilog", Matrix::row_vector(&log_lines.1));
    e.sample = Some(m.mask.clone());
    if let Some(sv) = &m.svy {
        crate::rpt_w6::mlsvy::post(&mut e, &me.engine.dataset, sv, &m.rows, &m.w);
    }
    let st = &mut me.engine.store;
    st.e = e;
    put(m);
    let mut out = log_lines.0;
    if !has("noheader", 5) || !has("notable", 5) {
        out.push(display(me, &opts)?);
    }
    Ok(out.join("\n"))
}

type MaxOut = ((Vec<String>, Vec<f64>), f64, Matrix, Option<Vec<Vec<f64>>>, bool);

fn run_max(me: &mut AdoInterpreter, m: &mut Model, iterate: usize, nolog: bool) -> Result<MaxOut, String> {
    let mut lines = Vec::new();
    let mut ll = search_values(me, m, &mut lines)?;
    let mut log = Vec::new();
    let mut converged = false;
    let k = m.b.len();
    let mut it = 0;
    loop {
        lines.push(format!("Iteration {it}:{}Log likelihood = {}", if it < 10 { "  " } else { " " }, fmt_g(ll, 10)));
        log.push(ll);
        let (g, h, _) = derivs(me, m, &m.b.clone(), ll)?;
        if it >= iterate {
            break;
        }
        // Newton direction (fall back to steepest ascent when not concave)
        let concave = h.cholesky().is_ok();
        if !concave {
            if let Some(l) = lines.last_mut() {
                l.push_str("  (not concave)");
            }
        }
        let step: Vec<f64> = if concave {
            let (hinv, _) = scaled_inverse(&h);
            (0..k).map(|i| (0..k).map(|j| hinv[(i, j)] * g[j]).sum()).collect()
        } else {
            // not concave here: ridge the (unit-diagonal) information matrix
            let d: Vec<f64> = (0..k).map(|i| { let x = h[(i, i)].abs().sqrt(); if x > 0.0 && x.is_finite() { x } else { 1.0 } }).collect();
            let mut found = None;
            for lam in [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e4] {
                let mut hs = h.clone();
                for i in 0..k {
                    hs[(i, i)] += lam * d[i] * d[i];
                }
                if hs.cholesky().is_ok() {
                    let (hinv, _) = scaled_inverse(&hs);
                    found = Some((0..k).map(|i| (0..k).map(|j| hinv[(i, j)] * g[j]).sum()).collect());
                    break;
                }
            }
            found.unwrap_or_else(|| (0..k).map(|i| g[i] / (d[i] * d[i])).collect())
        };
        let gnorm: f64 = (0..k).map(|i| g[i] * step[i]).sum::<f64>().abs();
        let mut lambda = 1.0;
        let mut accepted = None;
        for _ in 0..60 {
            let nb: Vec<f64> = m.b.iter().zip(&step).map(|(b, s)| b + lambda * s).collect();
            if let Some(v) = loglik(me, m, &nb)? {
                if v >= ll - 1e-12 * ll.abs().max(1.0) {
                    accepted = Some((nb, v));
                    break;
                }
            }
            lambda /= 2.0;
        }
        let Some((nb, v)) = accepted else { break };
        let dll = (v - ll).abs();
        let db = m.b.iter().zip(&nb).map(|(a, c)| (a - c).abs() / (a.abs() + 1.0)).fold(0.0, f64::max);
        m.b = nb;
        ll = v;
        it += 1;
        if concave && (gnorm < 1e-9 || (gnorm < 1e-5 && dll < 1e-15 * ll.abs().max(1.0) && db < 1e-12)) {
            lines.push(format!("Iteration {it}:{}Log likelihood = {}", if it < 10 { "  " } else { " " }, fmt_g(ll, 10)));
            log.push(ll);
            converged = true;
            break;
        }
    }
    let (_, h, scores) = derivs(me, m, &m.b.clone(), ll)?;
    if nolog {
        lines.clear();
    }
    if !converged {
        lines.push("convergence not achieved".into());
    }
    Ok(((lines, log), ll, h, scores, converged))
}

fn display(me: &mut AdoInterpreter, opts: &[(String, Option<String>)]) -> Result<String, String> {
    let e = me.engine.store.e.clone();
    let b = e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let v = e.matrices.get("V").ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let level = match opts.iter().find(|(o, _)| o.len() >= 1 && "level".starts_with(o.to_lowercase().as_str())).and_then(|(_, a)| a.clone()) {
        Some(l) => l.trim().parse::<f64>().map_err(|_| rc_err(198, "level() invalid"))?,
        None => me.engine.store.level,
    };
    let neq = opts.iter().find(|(o, _)| o == "neq").and_then(|(_, a)| a.as_deref().and_then(|s| s.trim().parse::<usize>().ok()));
    let eform = opts.iter().find(|(o, _)| o == "eform").and_then(|(_, a)| a.clone()).map(|s| util::unquote(&s));
    let n = e.get("N").unwrap_or(SYSMISS);
    let ll = e.get("ll").unwrap_or(SYSMISS);
    let mut out = vec![String::new()];
    let title = e.macros.get("title").cloned().unwrap_or_default();
    let right1 = format!("{:<13} = {:>10}", "Number of obs", util::commas(n as usize));
    out.push(format!("{:<51}{}", title, right1));
    let (l2, l3) = match (e.get("chi2"), e.get("df_m")) {
        (Some(c), Some(df)) => (format!("{:<13} = {:>10.2}", format!("Wald chi2({})", df), c), format!("{:<13} = {:>10.4}", "Prob > chi2", dist::chi2tail(df, c))),
        _ => (format!("{:<13} = {:>10}", "Wald chi2(0)", "."), format!("{:<13} = {:>10}", "Prob > chi2", ".")),
    };
    out.push(format!("{:<51}{}", "", l2));
    let left = format!("Log likelihood = {}", fmt_g(ll, 10));
    out.push(format!("{:<51}{}", left, l3));
    out.push(String::new());
    let mut names = b.colnames.clone();
    let k = names.len();
    // drop equations beyond neq()
    let keep: Vec<usize> = match neq {
        Some(q) => {
            let mut eqs: Vec<String> = Vec::new();
            for c in &names {
                let eq = c.split(':').next().unwrap_or("").to_string();
                if !eqs.contains(&eq) {
                    eqs.push(eq);
                }
            }
            (0..k).filter(|&j| eqs.iter().position(|q2| Some(q2.as_str()) == names[j].split(':').next()).unwrap_or(0) < q).collect()
        }
        None => (0..k).collect(),
    };
    // a single equation is shown without its name
    let single = names.iter().map(|c| c.split(':').next().unwrap_or("")).collect::<std::collections::BTreeSet<_>>().len() == 1;
    if single {
        names = names.iter().map(|c| c.split_once(':').map_or(c.clone(), |x| x.1.to_string())).collect();
    }
    let bs: Vec<f64> = keep.iter().map(|&j| b.data[j]).collect();
    let mut vs = Matrix::zeros(keep.len(), keep.len());
    for (a, &i) in keep.iter().enumerate() {
        for (c, &j) in keep.iter().enumerate() {
            vs[(a, c)] = v[(i, j)];
        }
    }
    let nm: Vec<String> = keep.iter().map(|&j| names[j].clone()).collect();
    let no = vec![false; nm.len()];
    let depvar = e.macros.get("depvar").cloned().unwrap_or_default();
    let dep1 = depvar.split_whitespace().next().unwrap_or("").to_string();
    let spec = TableSpec { depvar: &dep1, names: &nm, b: &bs, v: &vs, omitted: &no, is_base: &no, df: None, level, eform: eform.as_deref(), coef_title: "Coefficient" };
    let mut t = est::coef_table(&spec);
    if e.macros.get("vcetype").is_some_and(|s| s == "Robust") && t.len() > 1 {
        let bar = t[1].find('|').unwrap_or(13);
        t.insert(1, format!("{}|{:>23}", " ".repeat(bar), "Robust"));
    }
    out.extend(t);
    Ok(out.join("\n"))
}

fn query() -> Result<String, String> {
    let m = get()?;
    let mut out = vec![String::new(), format!("Method:            {}", m.method), format!("Program:           {}", m.prog)];
    let deps: Vec<String> = m.eqs.iter().flat_map(|e| e.depvars.iter().cloned()).collect();
    out.push(format!("Dep. variable:     {}", deps.join(" ")));
    out.push(format!("{} equation{}:", m.eqs.len(), if m.eqs.len() == 1 { "" } else { "s" }));
    for e in &m.eqs {
        out.push(format!("  {:<14} {}{}", e.name, e.indep.join(" "), if e.cons { "" } else { " (noconstant)" }));
    }
    out.push(format!("Current (initial) values:\n  {}", m.b.iter().zip(&m.names).map(|(b, n)| format!("{n} = {}", fmt_g(*b, 10))).collect::<Vec<_>>().join("\n  ")));
    Ok(out.join("\n"))
}

fn plot(me: &mut AdoInterpreter, args: &str) -> Result<String, String> {
    let mut m = get()?;
    let words: Vec<&str> = args.split(',').next().unwrap_or("").split_whitespace().collect();
    let name = words.first().ok_or_else(|| rc_err(198, "ml plot [eqname:]name [# [# [#]]]"))?;
    let full = if name.contains(':') { name.to_string() } else { format!("{}:{}", m.eqs[0].name, name) };
    let j = m.names.iter().position(|c| *c == full || c == &format!("{name}:_cons")).ok_or_else(|| rc_err(111, format!("{name} not found in model")))?;
    let nums: Vec<f64> = words[1..].iter().filter_map(|w| w.parse().ok()).collect();
    let b0 = m.b[j];
    let (lo, hi) = match nums.as_slice() {
        [a] => (b0 - a, b0 + a),
        [a, c, ..] => (*a, *c),
        _ => (b0 - 1.0, b0 + 1.0),
    };
    setup(me, &m)?;
    let mut pts = Vec::new();
    let mut best = (b0, f64::NEG_INFINITY);
    for s in 0..=20 {
        let x = lo + (hi - lo) * s as f64 / 20.0;
        let mut b = m.b.clone();
        b[j] = x;
        if let Some(v) = loglik(me, &m, &b)? {
            pts.push((x, v));
            if v > best.1 {
                best = (x, v);
            }
        }
    }
    cleanup(me, &m);
    if best.1.is_finite() {
        m.b[j] = best.0;
    }
    put(m);
    let mut g = ferrostat_cmd::graph::Graph {
        layers: vec![ferrostat_cmd::graph::Layer { kind: ferrostat_cmd::graph::LayerKind::Connected, points: pts, label: "ln L()".into(), ..Default::default() }],
        xtitle: full.clone(),
        ytitle: "ln L()".into(),
        ..Default::default()
    };
    g.name = "Graph".into();
    me.engine.store.graphs.insert(g.name.clone(), g.clone());
    me.engine.store.graph = Some(g);
    Ok(format!("(reset {full} = {}; log likelihood = {})", fmt_g(best.0, 10), fmt_g(best.1, 10)))
}

fn score(me: &mut AdoInterpreter, args: &str) -> Result<String, String> {
    let m = get()?;
    if !is_obs_method(&m.method) {
        return Err(rc_err(198, "ml score requires an observation-level method"));
    }
    let names: Vec<String> = args.split(',').next().unwrap_or("").split_whitespace().map(String::from).collect();
    let stub = names.len() == 1 && names[0].ends_with('*');
    let neq = m.eqs.len();
    let newnames: Vec<String> = if stub { (1..=neq).map(|i| format!("{}{i}", names[0].trim_end_matches('*'))).collect() } else { names };
    if newnames.len() != neq {
        return Err(rc_err(198, format!("{neq} new variable names required")));
    }
    let b = me.engine.store.e.matrices.get("b").map(|x| x.data.clone()).unwrap_or(m.b.clone());
    setup(me, &m)?;
    let th = thetas(&m, &b);
    let n = m.rows.len();
    let mut cols = Vec::new();
    for j in 0..neq {
        let h: Vec<f64> = th[j].iter().map(|x| 1e-4 * (x.abs() + 1.0)).collect();
        let mut tu = th.clone();
        let mut td = th.clone();
        for i in 0..n {
            tu[j][i] += h[i];
            td[j][i] -= h[i];
        }
        let u = lnf_obs(me, &m, &tu, &b)?.ok_or_else(|| rc_err(1400, "could not compute scores"))?;
        let d = lnf_obs(me, &m, &td, &b)?.ok_or_else(|| rc_err(1400, "could not compute scores"))?;
        cols.push((0..n).map(|i| (u[i] - d[i]) / (2.0 * h[i])).collect::<Vec<f64>>());
    }
    cleanup(me, &m);
    for (j, name) in newnames.iter().enumerate() {
        varlist::check_name(name)?;
        if me.engine.dataset.var_index.contains_key(name) {
            return Err(rc_err(110, format!("variable {name} already defined")));
        }
        set_var(me, name, &m, Some(&cols[j]))?;
    }
    Ok(String::new())
}

// ---------------------------------------------------------------------------
// Evaluator helpers
// ---------------------------------------------------------------------------

fn eq_index(opts: &[(String, Option<String>)], neq: usize) -> Result<Vec<usize>, String> {
    match opts.iter().find(|(o, _)| o.to_lowercase() == "eq").and_then(|(_, a)| a.clone()) {
        Some(a) => a
            .split(',')
            .map(|s| {
                let s = s.trim().trim_start_matches('#');
                s.parse::<usize>().map(|x| x - 1).map_err(|_| rc_err(198, "eq() invalid"))
            })
            .collect::<Result<Vec<_>, _>>()
            .and_then(|v| if v.iter().all(|&x| x < neq) { Ok(v) } else { Err(rc_err(198, "eq() out of range")) }),
        None => Ok(vec![0]),
    }
}

fn mleval(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    let m = get()?;
    let (body, opts) = match find_top_level(rest, ',') {
        Some(i) => (rest[..i].trim(), split_options(&rest[i + 1..])),
        None => (rest.trim(), Vec::new()),
    };
    let (lhs, rhs) = body.split_once('=').ok_or_else(|| rc_err(198, "mleval newvar = vecname"))?;
    let lhs = lhs.trim().to_string();
    let bname = rhs.trim();
    let bm = me.engine.store.matrices.get(bname).cloned().ok_or_else(|| rc_err(111, format!("matrix {bname} not found")))?;
    let j = eq_index(&opts, m.eqs.len())?[0];
    let th = thetas(&m, &bm.data);
    let scalar = opts.iter().any(|(o, _)| o == "scalar");
    if scalar {
        if !m.x[j].is_empty() && !(m.x[j].len() == 1 && m.eqs[j].cons && m.eqs[j].indep.is_empty()) {
            return Err(rc_err(198, "mleval: scalar option requires a constant-only equation"));
        }
        me.engine.store.scalars.insert(lhs, th[j].first().copied().unwrap_or(SYSMISS));
        return Ok(());
    }
    let _ = me.engine.dataset.drop_variable(&lhs);
    set_var(me, &lhs, &m, Some(&th[j]))
}

/// Sum of an expression over the sample (weighted); missing if any value is.
fn sum_expr(me: &AdoInterpreter, m: &Model, text: &str, weight_of: impl Fn(usize) -> f64) -> Result<f64, String> {
    let e = parse_expression(text).map_err(|er| rc_err(198, er.message))?;
    let env = Env::with_store(&me.engine.dataset, &me.engine.store);
    let mut s = 0.0;
    for (t, &r) in m.rows.iter().enumerate() {
        let v = env.eval(&e, r)?.num()?;
        if is_missing(v) {
            return Ok(SYSMISS);
        }
        s += m.w[t] * v * weight_of(t);
    }
    Ok(s)
}

fn mlsum(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    let m = get()?;
    let (lhs, rhs) = rest.split_once('=').ok_or_else(|| rc_err(198, "mlsum scalarname = exp"))?;
    let s = sum_expr(me, &m, rhs.trim(), |_| 1.0)?;
    me.engine.store.scalars.insert(lhs.trim().to_string(), s);
    Ok(())
}

fn mlvecsum(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    let m = get()?;
    let (body, opts) = match find_top_level(rest, ',') {
        Some(i) => (rest[..i].trim(), split_options(&rest[i + 1..])),
        None => (rest.trim(), Vec::new()),
    };
    let (lhs, rhs) = body.split_once('=').ok_or_else(|| rc_err(198, "mlvecsum scalarname_lnf rowvecname = exp"))?;
    let names: Vec<&str> = lhs.split_whitespace().collect();
    let (lnf, vec_name) = (names.first().copied().unwrap_or(""), names.get(1).copied().unwrap_or(""));
    let j = eq_index(&opts, m.eqs.len())?[0];
    let mut out = Vec::new();
    for col in &m.x[j] {
        let col = col.clone();
        out.push(sum_expr(me, &m, rhs.trim(), |t| col[t])?);
    }
    if out.iter().any(|x| is_missing(*x)) {
        me.engine.store.scalars.insert(lnf.to_string(), SYSMISS);
    }
    let names: Vec<String> = m.names.iter().filter(|n| n.starts_with(&format!("{}:", m.eqs[j].name))).cloned().collect();
    me.engine.store.matrices.insert(vec_name.to_string(), Matrix::row_vector(&out).with_names(vec!["r1".into()], names));
    Ok(())
}

fn mlmatsum(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    let m = get()?;
    let (body, opts) = match find_top_level(rest, ',') {
        Some(i) => (rest[..i].trim(), split_options(&rest[i + 1..])),
        None => (rest.trim(), Vec::new()),
    };
    let (lhs, rhs) = body.split_once('=').ok_or_else(|| rc_err(198, "mlmatsum scalarname_lnf matrixname = exp"))?;
    let names: Vec<&str> = lhs.split_whitespace().collect();
    let (lnf, mat_name) = (names.first().copied().unwrap_or(""), names.get(1).copied().unwrap_or(""));
    let eqs = eq_index(&opts, m.eqs.len())?;
    let (ja, jb) = (eqs[0], *eqs.get(1).unwrap_or(&eqs[0]));
    let (ka, kb) = (m.x[ja].len(), m.x[jb].len());
    let mut out = Matrix::zeros(ka, kb);
    for a in 0..ka {
        for c in 0..kb {
            let (xa, xb) = (m.x[ja][a].clone(), m.x[jb][c].clone());
            out[(a, c)] = sum_expr(me, &m, rhs.trim(), |t| xa[t] * xb[t])?;
        }
    }
    if out.data.iter().any(|x| is_missing(*x)) {
        me.engine.store.scalars.insert(lnf.to_string(), SYSMISS);
    }
    me.engine.store.matrices.insert(mat_name.to_string(), out);
    Ok(())
}

/// Root mean square of each regressor (1 for constants).
fn col_scales(m: &Model) -> Vec<f64> {
    let mut out = Vec::new();
    for cols in &m.x {
        for c in cols {
            let s = (c.iter().map(|x| x * x).sum::<f64>() / c.len().max(1) as f64).sqrt();
            out.push(if s > 0.0 && s.is_finite() { s } else { 1.0 });
        }
    }
    out
}

/// Inverse of a symmetric matrix after scaling it to unit diagonal.
fn scaled_inverse(h: &Matrix) -> (Matrix, Vec<usize>) {
    let k = h.rows;
    let d: Vec<f64> = (0..k).map(|i| { let x = h[(i, i)].abs().sqrt(); if x > 0.0 && x.is_finite() { x } else { 1.0 } }).collect();
    let mut s = Matrix::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            s[(i, j)] = h[(i, j)] / (d[i] * d[j]);
        }
    }
    let (inv, dropped) = s.sym_inverse(1e-12);
    let mut out = Matrix::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            out[(i, j)] = inv[(i, j)] / (d[i] * d[j]);
        }
    }
    (out, dropped)
}
