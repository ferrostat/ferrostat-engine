//! Nonlinear mixed-effects regression (`menl`) and the `estat df`,
//! `estat sd` and `estat wcorrelation` reports after `mixed`.
//!
//! `menl depvar = <expression>` where the expression refers to scalar
//! parameters `{name}` (optionally with an initial value, `{name=1}`),
//! linear combinations `{name: varlist[, noconstant]}` and random effects
//! `{U[groupvar]}` (one grouping variable; several random effects are
//! independent with their own variances). Errors are Gaussian.
//!
//! The marginal likelihood is computed by the Laplace approximation around
//! the conditional modes of the random effects (found by Gauss-Newton on
//! the penalized least-squares criterion), which is the first-order
//! linearization of Lindstrom and Bates (1990) evaluated by maximum
//! likelihood; for models linear in the random effects it is exact, so a
//! linear `menl` reproduces `mixed` (ML). The variance of the estimates is
//! the inverse of the numerical observed information.

use std::collections::BTreeMap;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::{BinOp, Command, Expr, UnaryOp};

use crate::commands::CommandHandler;
use crate::dist;
use crate::est::{self, Likelihood};
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![("menl", Box::new(Menl))]
}

// ---------------------------------------------------------------------------
// expression compilation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Node {
    Num(f64),
    Col(usize),
    Par(usize),
    Re(usize),
    Bin(BinOp, Box<Node>, Box<Node>),
    Neg(Box<Node>),
    Not(Box<Node>),
    Call(String, Vec<Node>),
}

struct Ctx<'a> {
    cols: &'a [Vec<f64>],
    par: &'a [f64],
    re: &'a [f64],
}

fn ev(n: &Node, c: &Ctx, i: usize) -> f64 {
    match n {
        Node::Num(v) => *v,
        Node::Col(j) => c.cols[*j][i],
        Node::Par(j) => c.par[*j],
        Node::Re(j) => c.re[*j],
        Node::Neg(a) => -ev(a, c, i),
        Node::Not(a) => {
            if ev(a, c, i) == 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Node::Bin(op, a, b) => {
            let (x, y) = (ev(a, c, i), ev(b, c, i));
            let t = |v: bool| if v { 1.0 } else { 0.0 };
            match op {
                BinOp::Add => x + y,
                BinOp::Sub => x - y,
                BinOp::Mul => x * y,
                BinOp::Div => x / y,
                BinOp::Pow => x.powf(y),
                BinOp::Eq => t(x == y),
                BinOp::Ne => t(x != y),
                BinOp::Lt => t(x < y),
                BinOp::Le => t(x <= y),
                BinOp::Gt => t(x > y),
                BinOp::Ge => t(x >= y),
                BinOp::And => t(x != 0.0 && y != 0.0),
                BinOp::Or => t(x != 0.0 || y != 0.0),
            }
        }
        Node::Call(f, a) => {
            let v: Vec<f64> = a.iter().map(|x| ev(x, c, i)).collect();
            match (f.as_str(), v.len()) {
                ("exp", 1) => v[0].exp(),
                ("ln" | "log", 1) => v[0].ln(),
                ("log10", 1) => v[0].log10(),
                ("sqrt", 1) => v[0].sqrt(),
                ("abs", 1) => v[0].abs(),
                ("sin", 1) => v[0].sin(),
                ("cos", 1) => v[0].cos(),
                ("tan", 1) => v[0].tan(),
                ("atan", 1) => v[0].atan(),
                ("tanh", 1) => v[0].tanh(),
                ("invlogit", 1) => 1.0 / (1.0 + (-v[0]).exp()),
                ("logit", 1) => (v[0] / (1.0 - v[0])).ln(),
                ("normal", 1) => dist::normal(v[0]),
                ("normalden", 1) => dist::normalden(v[0]),
                ("min", _) => v.iter().cloned().fold(f64::INFINITY, f64::min),
                ("max", _) => v.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                ("cond", 3) => {
                    if v[0] != 0.0 {
                        v[1]
                    } else {
                        v[2]
                    }
                }
                _ => f64::NAN,
            }
        }
    }
}

struct Spec {
    /// fixed parameter names (display) and initial values
    pars: Vec<String>,
    init: Vec<f64>,
    /// random-effect names
    res: Vec<String>,
    group: String,
    vars: Vec<String>,
    node: Node,
}

const FUNCS: &[&str] = &["exp", "ln", "log", "log10", "sqrt", "abs", "sin", "cos", "tan", "atan", "tanh", "invlogit", "logit", "normal", "normalden", "min", "max", "cond"];

fn compile(e: &Expr, ds: &DataSet, sp: &mut Spec) -> Result<Node, String> {
    Ok(match e {
        Expr::Number(v) => Node::Num(*v),
        Expr::Paren(a) => compile(a, ds, sp)?,
        Expr::Unary(UnaryOp::Neg, a) => Node::Neg(Box::new(compile(a, ds, sp)?)),
        Expr::Unary(UnaryOp::Not, a) => Node::Not(Box::new(compile(a, ds, sp)?)),
        Expr::Binary(a, op, b) => Node::Bin(op.clone(), Box::new(compile(a, ds, sp)?), Box::new(compile(b, ds, sp)?)),
        Expr::Call(f, args) => {
            let f = f.to_lowercase();
            if !FUNCS.contains(&f.as_str()) {
                return Err(rc_err(133, format!("function {f}() not allowed in menl")));
            }
            Node::Call(f, args.iter().map(|a| compile(a, ds, sp)).collect::<Result<_, _>>()?)
        }
        Expr::Var(name) => {
            if let Some(p) = name.strip_prefix("__mp_") {
                let j = sp.pars.iter().position(|x| x == p).ok_or_else(|| rc_err(198, format!("parameter {p} unknown")))?;
                Node::Par(j)
            } else if let Some(r) = name.strip_prefix("__mr_") {
                let j = sp.res.iter().position(|x| x == r).ok_or_else(|| rc_err(198, format!("random effect {r} unknown")))?;
                Node::Re(j)
            } else {
                let v = crate::varlist::resolve_name(ds, name)?;
                let j = match sp.vars.iter().position(|x| *x == v) {
                    Some(j) => j,
                    None => {
                        sp.vars.push(v);
                        sp.vars.len() - 1
                    }
                };
                Node::Col(j)
            }
        }
        _ => return Err(rc_err(198, "invalid menl expression")),
    })
}

/// Replace the braces of a menl expression by identifiers and collect the
/// parameters and random effects.
fn preprocess(ds: &DataSet, text: &str) -> Result<(String, Spec), String> {
    let mut sp = Spec { pars: Vec::new(), init: Vec::new(), res: Vec::new(), group: String::new(), vars: Vec::new(), node: Node::Num(0.0) };
    let mut out = String::new();
    let mut rest = text;
    while let Some(i) = rest.find('{') {
        out.push_str(&rest[..i]);
        let j = rest[i..].find('}').ok_or_else(|| rc_err(198, "unmatched {"))? + i;
        let inner = rest[i + 1..j].trim();
        rest = &rest[j + 1..];
        if let Some((name, list)) = inner.split_once(':') {
            // linear combination
            let name = name.trim();
            let (vl, opts) = list.split_once(',').unwrap_or((list, ""));
            let vars = crate::varlist::expand(ds, vl.trim())?;
            let mut terms = Vec::new();
            for v in &vars {
                let pn = format!("{name}:{v}");
                if !sp.pars.contains(&pn) {
                    sp.pars.push(pn.clone());
                    sp.init.push(0.0);
                }
                terms.push(format!("__mp_{}*{v}", pn.replace(':', "\u{1}")));
            }
            if !opts.to_lowercase().contains("nocons") {
                let pn = format!("{name}:_cons");
                if !sp.pars.contains(&pn) {
                    sp.pars.push(pn.clone());
                    sp.init.push(0.0);
                }
                terms.push(format!("__mp_{}", pn.replace(':', "\u{1}")));
            }
            out.push_str(&format!("({})", terms.join("+")));
        } else if let Some(k) = inner.find('[') {
            let name = inner[..k].trim().to_string();
            let g = inner[k + 1..].trim_end_matches(']').trim();
            let g = crate::varlist::resolve_name(ds, g)?;
            if !sp.group.is_empty() && sp.group != g {
                return Err(rc_err(198, "only one grouping variable (one level) is supported"));
            }
            sp.group = g;
            if !sp.res.contains(&name) {
                sp.res.push(name.clone());
            }
            out.push_str(&format!("__mr_{name}"));
        } else {
            let (name, init) = match inner.split_once('=') {
                Some((a, b)) => (a.trim().to_string(), b.trim().parse::<f64>().map_err(|_| rc_err(198, format!("invalid initial value in {{{inner}}}")))?),
                None => (inner.to_string(), SYSMISS),
            };
            match sp.pars.iter().position(|x| *x == name) {
                Some(p) => {
                    if !is_missing(init) {
                        sp.init[p] = init;
                    }
                }
                None => {
                    sp.pars.push(name.clone());
                    sp.init.push(if is_missing(init) { 0.0 } else { init });
                }
            }
            out.push_str(&format!("__mp_{name}"));
        }
    }
    out.push_str(rest);
    Ok((out, sp))
}

// ---------------------------------------------------------------------------
// Laplace likelihood
// ---------------------------------------------------------------------------

struct Model<'a> {
    node: &'a Node,
    cols: &'a [Vec<f64>],
    y: &'a [f64],
    groups: &'a [Vec<usize>],
    kp: usize,
    q: usize,
}

impl Model<'_> {
    fn f(&self, par: &[f64], u: &[f64], i: usize) -> f64 {
        ev(self.node, &Ctx { cols: self.cols, par, re: u }, i)
    }
    /// Laplace log likelihood of one group and the conditional modes.
    fn group_ll(&self, par: &[f64], tau2: &[f64], s2: f64, g: &[usize]) -> (f64, Vec<f64>) {
        let q = self.q;
        let mut u = vec![0.0; q];
        let crit = |u: &[f64]| -> f64 {
            let mut s = 0.0;
            for &i in g {
                let r = self.y[i] - self.f(par, u, i);
                s += r * r;
            }
            s / s2 + (0..q).map(|k| u[k] * u[k] / tau2[k]).sum::<f64>()
        };
        let jac = |u: &[f64]| -> (Vec<f64>, Vec<Vec<f64>>) {
            let r: Vec<f64> = g.iter().map(|&i| self.y[i] - self.f(par, u, i)).collect();
            let j: Vec<Vec<f64>> = (0..q)
                .map(|k| {
                    let h = 1e-6 * u[k].abs().max(1.0);
                    let mut up = u.to_vec();
                    let mut um = u.to_vec();
                    up[k] += h;
                    um[k] -= h;
                    g.iter().map(|&i| (self.f(par, &up, i) - self.f(par, &um, i)) / (2.0 * h)).collect()
                })
                .collect();
            (r, j)
        };
        let mut cur = crit(&u);
        let mut last_j = Vec::new();
        for _ in 0..100 {
            let (r, j) = jac(&u);
            let mut a = Matrix::zeros(q, q);
            let mut rhs = vec![0.0; q];
            for k in 0..q {
                for l in 0..q {
                    a[(k, l)] = (0..g.len()).map(|t| j[k][t] * j[l][t]).sum::<f64>() / s2;
                }
                a[(k, k)] += 1.0 / tau2[k];
                rhs[k] = (0..g.len()).map(|t| j[k][t] * r[t]).sum::<f64>() / s2 - u[k] / tau2[k];
            }
            last_j = j;
            let (ai, _) = a.sym_inverse(1e-14);
            let step: Vec<f64> = (0..q).map(|k| (0..q).map(|l| ai[(k, l)] * rhs[l]).sum()).collect();
            let mut lam = 1.0;
            let mut moved = false;
            for _ in 0..30 {
                let nu: Vec<f64> = (0..q).map(|k| u[k] + lam * step[k]).collect();
                let c = crit(&nu);
                if c.is_finite() && c <= cur + 1e-14 * cur.abs() {
                    let done = (cur - c).abs() < 1e-13 * (1.0 + cur.abs()) && step.iter().all(|s| (lam * s).abs() < 1e-9);
                    u = nu;
                    cur = c;
                    moved = !done;
                    break;
                }
                lam /= 2.0;
            }
            if !moved {
                break;
            }
        }
        let (_, j) = if last_j.is_empty() { jac(&u) } else { jac(&u) };
        let mut a = Matrix::zeros(q, q);
        for k in 0..q {
            for l in 0..q {
                a[(k, l)] = (0..g.len()).map(|t| j[k][t] * j[l][t]).sum::<f64>() / s2;
            }
            a[(k, k)] += 1.0 / tau2[k];
        }
        let n = g.len() as f64;
        let ll = -0.5 * n * (2.0 * std::f64::consts::PI * s2).ln() - 0.5 * tau2.iter().map(|t| t.ln()).sum::<f64>() - 0.5 * cur - 0.5 * a.det().max(1e-300).ln();
        (ll, u)
    }
    fn split<'b>(&self, th: &'b [f64]) -> (&'b [f64], Vec<f64>, f64) {
        let par = &th[..self.kp];
        let tau2: Vec<f64> = (0..self.q).map(|k| (2.0 * th[self.kp + k]).exp()).collect();
        let s2 = (2.0 * th[self.kp + self.q]).exp();
        (par, tau2, s2)
    }
    fn ll(&self, th: &[f64]) -> f64 {
        let (par, tau2, s2) = self.split(th);
        let mut s = 0.0;
        for g in self.groups {
            let l = self.group_ll(par, &tau2, s2, g).0;
            if !l.is_finite() {
                return f64::NEG_INFINITY;
            }
            s += l;
        }
        s
    }
}

impl Likelihood for Model<'_> {
    fn k(&self) -> usize {
        self.kp + self.q + 1
    }
    fn eval(&self, b: &[f64], _need_scores: bool) -> (f64, Vec<f64>, Matrix, Vec<Vec<f64>>) {
        let (f0, g, h) = crate::zip::num_derivs(&|x: &[f64]| self.ll(x), b);
        let k = b.len();
        let mut info = Matrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                info[(i, j)] = -h[i][j];
            }
        }
        (f0, g, info, Vec::new())
    }
}

pub struct Menl;

impl CommandHandler for Menl {
    fn name(&self) -> &'static str {
        "menl"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        if main.is_empty() {
            return crate::models::replay(st, "menl");
        }
        // the main text can stop at a brace-internal comma; recover from raw
        let raw = cmd.raw.trim();
        let body = raw.split_once(char::is_whitespace).map(|x| x.1).unwrap_or("");
        let body = cut_clauses(body);
        let (lhs, rhs) = body.split_once('=').ok_or_else(|| rc_err(198, "menl requires depvar = <nonlinear expression>"))?;
        let dep = crate::varlist::resolve_name(ds, lhs.trim())?;
        let (text, mut sp) = preprocess(ds, rhs)?;
        let text = text.replace('\u{1}', "_C_");
        let pars_ident: Vec<String> = sp.pars.iter().map(|p| p.replace(':', "_C_")).collect();
        let ex = ferrostat_parser::parser::parse_expression(&text).map_err(|e| rc_err(198, e.message))?;
        let disp = sp.pars.clone();
        sp.pars = pars_ident;
        let node = compile(&ex, ds, &mut sp)?;
        sp.pars = disp;
        sp.node = node;
        if sp.res.is_empty() {
            return Err(rc_err(198, "no random effects specified; use {name[groupvar]}"));
        }
        // sample
        let mut mask = util::touse(ds, st, cmd)?;
        let mut need = sp.vars.clone();
        need.push(dep.clone());
        need.push(sp.group.clone());
        util::markout(ds, &mut mask, &need);
        let rows = util::rows(&mask);
        if rows.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        let cols: Vec<Vec<f64>> = sp.vars.iter().map(|v| util::col(ds, v).map(|c| rows.iter().map(|&r| c[r]).collect())).collect::<Result<_, _>>()?;
        let yall = util::col(ds, &dep)?;
        let y: Vec<f64> = rows.iter().map(|&r| yall[r]).collect();
        let gcol = util::col(ds, &sp.group)?;
        let mut gm: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        for (t, &r) in rows.iter().enumerate() {
            gm.entry((gcol[r] * 1e6).round() as i64).or_default().push(t);
        }
        let groups: Vec<Vec<usize>> = gm.into_values().collect();
        let kp = sp.pars.len();
        let q = sp.res.len();
        let model = Model { node: &sp.node, cols: &cols, y: &y, groups: &groups, kp, q };
        // starting values: nonlinear least squares ignoring the random effects
        let mut par = sp.init.clone();
        let zero = vec![0.0; q];
        let ssr = |p: &[f64]| -> f64 { (0..y.len()).map(|i| (y[i] - model.f(p, &zero, i)).powi(2)).sum() };
        let mut cur = ssr(&par);
        for _ in 0..200 {
            let r: Vec<f64> = (0..y.len()).map(|i| y[i] - model.f(&par, &zero, i)).collect();
            let j: Vec<Vec<f64>> = (0..kp)
                .map(|k| {
                    let h = 1e-6 * par[k].abs().max(1.0);
                    let mut pp = par.clone();
                    pp[k] += h;
                    (0..y.len()).map(|i| (model.f(&pp, &zero, i) - model.f(&par, &zero, i)) / h).collect()
                })
                .collect();
            let (b, _, _) = crate::xt_lin::ls(&j, &r);
            let mut lam = 1.0;
            let mut ok = false;
            for _ in 0..30 {
                let np: Vec<f64> = par.iter().zip(&b).map(|(a, d)| a + lam * d).collect();
                let c = ssr(&np);
                if c.is_finite() && c < cur {
                    ok = (cur - c) > 1e-12 * cur.max(1e-300);
                    par = np;
                    cur = c;
                    break;
                }
                lam /= 2.0;
            }
            if !ok {
                break;
            }
        }
        if !cur.is_finite() {
            return Err(rc_err(430, "could not find feasible initial values; specify {name=value}"));
        }
        let n = y.len() as f64;
        let s0 = (cur / n).sqrt().max(1e-6);
        let mut th = par.clone();
        th.extend(std::iter::repeat((0.5 * s0).ln()).take(q));
        th.push((0.7 * s0).ln());
        let res = crate::xmodels::maximize(&model, th, 300)?;
        let (vt, _) = res.info.sym_inverse(1e-14);
        let th = res.b.clone();
        let k = th.len();
        // natural-scale variances by the delta method
        let mut b = th.clone();
        let mut jd = vec![1.0; k];
        for j in kp..k {
            b[j] = (2.0 * th[j]).exp();
            jd[j] = 2.0 * b[j];
        }
        let mut v = Matrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                v[(i, j)] = jd[i] * vt[(i, j)] * jd[j];
            }
        }
        let mut names: Vec<String> = sp.pars.iter().map(|p| if p.contains(':') { p.clone() } else { format!("/:{p}") }).collect();
        for r in &sp.res {
            names.push(format!("/:var({r})"));
        }
        names.push("/:var(e)".into());
        let level = util::level(cmd, st)?;
        let crit = crate::choice_mixed::zcrit(level);
        let ng = groups.len();
        let sizes: Vec<usize> = groups.iter().map(|g| g.len()).collect();
        let (gmin, gmax) = (*sizes.iter().min().unwrap(), *sizes.iter().max().unwrap());
        let gavg = n / ng as f64;
        let mut out = est::iteration_log(&res.iterations, "Log likelihood");
        out.push(String::new());
        out.push(crate::choice_mixed::hrow("Mixed-effects ML nonlinear regression", "Number of obs", util::commas(y.len())));
        out.push(crate::choice_mixed::hrow(&format!("Group variable: {}", sp.group), "Number of groups", util::commas(ng)));
        out.push(format!("{:<48}Obs per group:", ""));
        out.push(format!("{:>65} = {:>10}", "min", util::commas(gmin)));
        out.push(format!("{:>65} = {:>10}", "avg", format!("{gavg:.1}")));
        out.push(format!("{:>65} = {:>10}", "max", util::commas(gmax)));
        out.push(String::new());
        out.push(format!("Log likelihood = {}", fmt_g(res.ll, 9)));
        out.push(String::new());
        // fixed effects table
        let disp: Vec<String> = sp.pars.clone();
        let mut tnames: Vec<String> = Vec::new();
        for p in &disp {
            tnames.push(if p.contains(':') { p.clone() } else { format!("/{p}") });
        }
        let vfix = {
            let mut m = Matrix::zeros(kp, kp);
            for i in 0..kp {
                for j in 0..kp {
                    m[(i, j)] = v[(i, j)];
                }
            }
            m
        };
        out.extend(est::coef_table(&est::TableSpec { depvar: &dep, names: &tnames, b: &b[..kp], v: &vfix, omitted: &vec![false; kp], is_base: &vec![false; kp], df: None, level, eform: None, coef_title: "Coefficient" }));
        out.push(String::new());
        out.push("-".repeat(78));
        out.push(format!("  Random-effects parameters  |   Estimate   Std. err.     [{}% conf. interval]", fmt_g(level, 4)));
        out.push(format!("{}+{}", "-".repeat(29), "-".repeat(48)));
        out.push(format!("{:<29}|", format!("{}: Independent", sp.group)));
        let rerow = |label: &str, j: usize| -> String {
            let se = v[(j, j)].max(0.0).sqrt();
            let sl = vt[(j, j)].max(0.0).sqrt() * 2.0;
            let lo = (2.0 * th[j] - crit * sl).exp();
            let hi = (2.0 * th[j] + crit * sl).exp();
            format!("{:>28} |  {:>9}  {:>9}     {:>9}   {:>9}", label, fmt_g(b[j], 8), fmt_g(se, 8), fmt_g(lo, 8), fmt_g(hi, 8))
        };
        for (r, name) in sp.res.iter().enumerate() {
            out.push(rerow(&format!("var({name})"), kp + r));
        }
        out.push(format!("{}+{}", "-".repeat(29), "-".repeat(48)));
        out.push(rerow("var(Residual)", k - 1));
        out.push("-".repeat(78));
        if !res.converged {
            out.push("convergence not achieved".into());
        }
        let text = out.join("\n");
        let mut emask = vec![false; ds.n_obs];
        for &r in &rows {
            emask[r] = true;
        }
        let e = st.begin_e();
        e.set("N", n);
        e.set("N_g", ng as f64);
        e.set("ll", res.ll);
        e.set("k", k as f64);
        e.set("k_f", kp as f64);
        e.set("k_r", (q + 1) as f64);
        e.set("converged", if res.converged { 1.0 } else { 0.0 });
        e.set("sigma_e", b[k - 1].sqrt());
        e.set_str("cmd", "menl");
        e.set_str("depvar", dep.clone());
        e.set_str("ivars", sp.group.clone());
        e.set_str("revars", sp.res.join(" "));
        e.set_str("method", "ML");
        e.set_str("title", "Mixed-effects ML nonlinear regression");
        e.set_str("expression", rhs.trim().to_string());
        e.set_str("properties", "b V");
        e.set_str("_output", text.clone());
        e.set_mat("b", Matrix::row_vector(&b).with_names(vec!["y1".into()], names.clone()));
        e.set_mat("V", v.with_names(names.clone(), names));
        e.sample = Some(emask);
        Ok(text)
    }
}

/// Body of the command without `if`/`in`/options (braces may contain commas).
fn cut_clauses(body: &str) -> String {
    let mut depth = 0i32;
    let chars: Vec<char> = body.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '{' | '(' | '[' => depth += 1,
            '}' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => return chars[..i].iter().collect(),
            _ => {}
        }
        if depth == 0 && chars[i].is_whitespace() {
            let tail: String = chars[i + 1..].iter().collect();
            let w = tail.split_whitespace().next().unwrap_or("");
            if w == "if" || w == "in" {
                return chars[..i].iter().collect();
            }
        }
        i += 1;
    }
    body.to_string()
}

// ---------------------------------------------------------------------------
// estat df / sd / wcorrelation after mixed
// ---------------------------------------------------------------------------

pub fn estat_mixed(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    if ecmd != "mixed" {
        return None;
    }
    match sub {
        "df" => Some(estat_df(st)),
        "sd" => Some(estat_sd(st)),
        "wcorrelation" | "wcor" | "wcorr" => Some(wcorrelation(ds, st, cmd)),
        _ => None,
    }
}

fn estat_df(st: &mut Store) -> Result<String, String> {
    let e = st.e.clone();
    let b = e.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "last estimates not found"))?;
    let kf = e.get("k_f").unwrap_or(b.cols as f64) as usize;
    let n = e.get("N").unwrap_or(SYSMISS);
    let rank = (0..kf).filter(|&j| b.data[j] != 0.0).count() as f64;
    let method = e.macros.get("dfmethod").cloned().unwrap_or_else(|| "residual".into());
    let given = e.matrices.get("df");
    let df: Vec<f64> = (0..kf).map(|j| given.map(|m| m.data[j]).unwrap_or(n - rank)).collect();
    let mut out = vec![String::new(), format!("Degrees of freedom: {}", if given.is_some() { method.as_str() } else { "residual" }), String::new(), "-".repeat(30), format!("{:>12} |{:>15}", "", "DF"), format!("{}+{}", "-".repeat(13), "-".repeat(16))];
    for j in 0..kf {
        let nm = b.colnames[j].rsplit(':').next().unwrap_or("").to_string();
        out.push(format!("{:>12} |{:>15}", crate::summarize::abbrev(&nm, 12), fmt_g(df[j], 9)));
    }
    out.push("-".repeat(30));
    let mut r = Results::new();
    let names: Vec<String> = b.colnames[..kf].to_vec();
    r.set_mat("df", Matrix::row_vector(&df).with_names(vec!["df".into()], names));
    st.r = r;
    Ok(out.join("\n"))
}

fn estat_sd(st: &mut Store) -> Result<String, String> {
    let e = st.e.clone();
    let ivars: Vec<String> = e.macros.get("ivars").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let respec: Vec<Vec<String>> = e.macros.get("_respec").map(|s| s.split(';').map(|x| x.split_whitespace().map(String::from).collect()).collect()).unwrap_or_default();
    let mut out = vec![String::new(), "-".repeat(46), format!("{:>29} |   Estimate", "Random-effects parameters"), format!("{}+{}", "-".repeat(30), "-".repeat(15))];
    let mut names = Vec::new();
    let mut vals = Vec::new();
    for (l, iv) in ivars.iter().enumerate() {
        let ps = e.matrices.get(&format!("_Psi{}", l + 1)).cloned().unwrap_or_else(|| Matrix::zeros(0, 0));
        let zn = respec.get(l).cloned().unwrap_or_default();
        out.push(format!("{:<30}|", format!("{iv}: {}", if ps.rows > 1 { "Unstructured" } else { "Identity" })));
        for r in 0..ps.rows.min(zn.len()) {
            let sd = ps[(r, r)].max(0.0).sqrt();
            let lab = format!("sd({})", zn[r]);
            out.push(format!("{:>29} |  {:>9}", lab, fmt_g(sd, 8)));
            names.push(format!("{iv}:{lab}"));
            vals.push(sd);
        }
        for r in 0..ps.rows.min(zn.len()) {
            for c in 0..r {
                let d = (ps[(r, r)] * ps[(c, c)]).sqrt();
                if ps[(r, c)] == 0.0 || d == 0.0 {
                    continue;
                }
                let lab = format!("corr({},{})", zn[r], zn[c]);
                out.push(format!("{:>29} |  {:>9}", crate::summarize::abbrev(&lab, 29), fmt_g(ps[(r, c)] / d, 8)));
                names.push(format!("{iv}:{lab}"));
                vals.push(ps[(r, c)] / d);
            }
        }
        out.push(format!("{}+{}", "-".repeat(30), "-".repeat(15)));
    }
    let se = e.get("sigma_e").unwrap_or(SYSMISS);
    out.push(format!("{:>29} |  {:>9}", "sd(Residual)", fmt_g(se, 8)));
    out.push("-".repeat(46));
    names.push("sd(Residual)".into());
    vals.push(se);
    let mut r = Results::new();
    r.set_mat("b", Matrix::row_vector(&vals).with_names(vec!["y1".into()], names));
    st.r = r;
    Ok(out.join("\n"))
}

fn wcorrelation(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let e = st.e.clone();
    let ivars: Vec<String> = e.macros.get("ivars").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let respec: Vec<Vec<String>> = e.macros.get("_respec").map(|s| s.split(';').map(|x| x.split_whitespace().map(String::from).collect()).collect()).unwrap_or_default();
    if ivars.is_empty() {
        return Err(rc_err(321, "no random effects"));
    }
    let sample = e.sample.clone().ok_or_else(|| rc_err(301, "estimation sample not found"))?;
    let gcols: Vec<Vec<f64>> = ivars.iter().map(|v| util::col(ds, v)).collect::<Result<_, _>>()?;
    let top = &gcols[0];
    let rows_all: Vec<usize> = (0..ds.n_obs).filter(|&i| sample.get(i).copied().unwrap_or(false)).collect();
    if rows_all.is_empty() {
        return Err(rc_err(2000, "no observations"));
    }
    let target = match cmd.opt("at", 2) {
        Some(a) => {
            let (v, val) = a.split_once('=').ok_or_else(|| rc_err(198, "at() requires groupvar = value"))?;
            let v = crate::varlist::resolve_name(ds, v.trim())?;
            if v != ivars[0] {
                return Err(rc_err(198, format!("at() must refer to the top-level group variable {}", ivars[0])));
            }
            val.trim().parse::<f64>().map_err(|_| rc_err(198, "at() invalid"))?
        }
        None => top[rows_all[0]],
    };
    let rows: Vec<usize> = rows_all.iter().copied().filter(|&i| top[i] == target).collect();
    if rows.is_empty() {
        return Err(rc_err(2000, "no observations in the requested group"));
    }
    let m = rows.len();
    let s2 = e.get("sigma_e").unwrap_or(1.0).powi(2);
    let mut cov = Matrix::zeros(m, m);
    for i in 0..m {
        cov[(i, i)] = s2;
    }
    for (l, zn) in respec.iter().enumerate().take(ivars.len()) {
        let ps = e.matrices.get(&format!("_Psi{}", l + 1)).cloned().unwrap_or_else(|| Matrix::zeros(0, 0));
        let zc: Vec<Vec<f64>> = zn.iter().map(|z| if z == "_cons" { Ok(vec![1.0; ds.n_obs]) } else { util::col(ds, z) }).collect::<Result<_, _>>()?;
        for a in 0..m {
            for c in 0..m {
                let same = (0..=l).all(|q| gcols[q][rows[a]] == gcols[q][rows[c]]);
                if !same {
                    continue;
                }
                let mut s = 0.0;
                for r1 in 0..ps.rows.min(zc.len()) {
                    for r2 in 0..ps.cols.min(zc.len()) {
                        s += zc[r1][rows[a]] * ps[(r1, r2)] * zc[r2][rows[c]];
                    }
                }
                cov[(a, c)] += s;
            }
        }
    }
    let mut corr = Matrix::zeros(m, m);
    for a in 0..m {
        for c in 0..m {
            corr[(a, c)] = cov[(a, c)] / (cov[(a, a)] * cov[(c, c)]).sqrt();
        }
    }
    let show_cov = cmd.has_opt("covariance", 3);
    let shown = if show_cov { &cov } else { &corr };
    let labels: Vec<String> = (1..=m).map(|i| format!("obs{i}")).collect();
    let mut out = vec![String::new(), format!("Standard deviations and correlations for {} = {}:", ivars[0], fmt_g(target, 9)), String::new()];
    if show_cov {
        out[1] = format!("Covariances for {} = {}:", ivars[0], fmt_g(target, 9));
    } else {
        out.push("Standard deviations:".into());
        out.push(format!("  {}", (0..m).map(|i| format!("{:>9}", fmt_g(cov[(i, i)].sqrt(), 7))).collect::<Vec<_>>().join(" ")));
        out.push(String::new());
        out.push("Correlations:".into());
    }
    out.push(format!("{:>8} |{}", "", labels.iter().map(|l| format!(" {:>9}", l)).collect::<String>()));
    out.push(format!("{}+{}", "-".repeat(9), "-".repeat(10 * m)));
    for a in 0..m {
        out.push(format!("{:>8} |{}", labels[a], (0..=a).map(|c| format!(" {:>9}", fmt_g(shown[(a, c)], 7))).collect::<String>()));
    }
    let mut r = Results::new();
    r.set_mat("Cov", cov.with_names(labels.clone(), labels.clone()));
    r.set_mat("Corr", corr.with_names(labels.clone(), labels));
    st.r = r;
    Ok(out.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clause_cutting_respects_braces() {
        assert_eq!(cut_clauses("y = {xb: x z, nocons} + {U[id]}, stddev"), "y = {xb: x z, nocons} + {U[id]}");
        assert_eq!(cut_clauses("y = {b0}*exp(x) if x > 0"), "y = {b0}*exp(x)");
    }
}
