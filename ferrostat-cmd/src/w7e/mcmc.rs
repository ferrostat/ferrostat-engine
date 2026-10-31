//! Adaptive random-walk Metropolis-Hastings with Gibbs updates for
//! conjugate blocks.
//!
//! Parameters with bounded support are sampled on an unconstrained scale
//! (log or logit) with the Jacobian included, so proposals never leave the
//! support. Each block has a multivariate normal proposal whose scale and
//! covariance adapt during burn-in toward a target acceptance rate
//! (0.44 for one parameter, decreasing to 0.234 for five or more).

use crate::matrix::Matrix;
use crate::rng::Rng;

use super::model::{Arg, Fam, Model, PK};
use super::prior::{Support, Uni};

#[derive(Clone, Debug)]
pub struct Settings {
    pub mcmcsize: usize,
    pub burnin: usize,
    pub thin: usize,
    pub every: usize,
    pub maxiter: usize,
    pub tarate: Option<f64>,
    pub nchains: usize,
    /// user blocks: params, gibbs requested, split into singletons
    pub blocks: Vec<(Vec<usize>, bool, bool)>,
    pub gibbs_all: bool,
    /// skip mode search
    pub nomode: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { mcmcsize: 10000, burnin: 2500, thin: 1, every: 100, maxiter: usize::MAX, tarate: None, nchains: 1, blocks: Vec::new(), gibbs_all: false, nomode: false }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    Mh,
    /// discrete parameter: random-walk on integers
    Discrete,
    /// Dirichlet-distributed parameters: random walk on additive log ratios
    Simplex,
    /// coefficients of a normal linear equation: (eq, likelihood, prior means, prior variances)
    GibbsCoef(usize, usize, Vec<f64>, Vec<f64>),
    /// variance of a normal likelihood with inverse-gamma(a, b) prior
    GibbsIgLik(usize, f64, f64),
    /// variance of normal priors on `members` with inverse-gamma(a, b) prior
    GibbsIgPrior(Vec<(usize, usize)>, f64, f64),
    /// random effects of a normal likelihood: (term, likelihood)
    GibbsRe(usize, usize),
    /// covariance matrix with inverse-Wishart(df, V) prior: (matrix, df, V)
    GibbsIw(usize, f64, Matrix),
}

#[derive(Clone, Debug)]
pub struct Block {
    pub idx: Vec<usize>,
    pub kind: Kind,
    pub chol: Matrix,
    pub lambda: f64,
    pub acc: usize,
    pub tries: usize,
    win_acc: usize,
    win_tries: usize,
    hist: Vec<Vec<f64>>,
    nadapt: usize,
}

pub struct Sim {
    /// kept draws on the original scale: rows = draws (all chains stacked)
    pub draws: Vec<Vec<f64>>,
    pub ll: Vec<f64>,
    pub lp: Vec<f64>,
    pub chain: Vec<usize>,
    pub arate: Vec<f64>,
    pub blocks: Vec<Block>,
    pub any_gibbs: bool,
    pub any_mh: bool,
    pub start: Vec<f64>,
}

// ---------------------------------------------------------------------------
// transformations
// ---------------------------------------------------------------------------

fn to_x(s: Support, u: f64) -> f64 {
    match s {
        Support::Real | Support::Discrete(..) => u,
        Support::Lower(lo) => lo + u.exp(),
        Support::Interval(a, b) => a + (b - a) / (1.0 + (-u).exp()),
    }
}

fn to_u(s: Support, x: f64) -> f64 {
    match s {
        Support::Real | Support::Discrete(..) => x,
        Support::Lower(lo) => (x - lo).max(1e-300).ln(),
        Support::Interval(a, b) => {
            let p = ((x - a) / (b - a)).clamp(1e-12, 1.0 - 1e-12);
            (p / (1.0 - p)).ln()
        }
    }
}

fn log_jac(s: Support, u: f64) -> f64 {
    match s {
        Support::Real | Support::Discrete(..) => 0.0,
        Support::Lower(_) => u,
        Support::Interval(a, b) => {
            let l = -super::model::ln1pexp(-u);
            let m = -super::model::ln1pexp(u);
            (b - a).ln() + l + m
        }
    }
}

/// Target acceptance rate for a block of dimension d.
fn target(d: usize, user: Option<f64>) -> f64 {
    if let Some(t) = user {
        return t;
    }
    match d {
        1 => 0.44,
        2 => 0.352,
        3 => 0.316,
        4 => 0.279,
        _ => 0.234,
    }
}

// ---------------------------------------------------------------------------
// mode search (Newton with numerical derivatives on the unconstrained scale)
// ---------------------------------------------------------------------------

fn post_u(m: &Model, th: &mut [f64], free: &[usize], u: &[f64]) -> f64 {
    let mut jac = 0.0;
    for (k, &j) in free.iter().enumerate() {
        let s = m.params[j].support;
        th[j] = to_x(s, u[k]);
        jac += log_jac(s, u[k]);
    }
    let v = m.log_post(th);
    if v.is_finite() { v + jac } else { f64::NEG_INFINITY }
}

/// Returns the negative inverse Hessian at the mode (u scale), when available.
pub fn find_mode(m: &Model, th: &mut Vec<f64>, free: &[usize]) -> Option<Matrix> {
    let k = free.len();
    if k == 0 || k > 80 {
        return None;
    }
    let mut u: Vec<f64> = free.iter().map(|&j| to_u(m.params[j].support, th[j])).collect();
    let u_start = u.clone();
    let mut work = th.clone();
    let mut f0 = post_u(m, &mut work, free, &u);
    if !f0.is_finite() {
        return None;
    }
    let mut hinv = None;
    for _it in 0..60 {
        let h = 1e-4;
        let mut g = vec![0.0; k];
        let mut hess = Matrix::zeros(k, k);
        let steps: Vec<f64> = u.iter().map(|v| h * v.abs().max(1.0)).collect();
        let mut fp = vec![0.0; k];
        let mut fm = vec![0.0; k];
        for i in 0..k {
            let mut up = u.clone();
            up[i] += steps[i];
            fp[i] = post_u(m, &mut work, free, &up);
            let mut um = u.clone();
            um[i] -= steps[i];
            fm[i] = post_u(m, &mut work, free, &um);
            g[i] = (fp[i] - fm[i]) / (2.0 * steps[i]);
            hess[(i, i)] = (fp[i] - 2.0 * f0 + fm[i]) / (steps[i] * steps[i]);
        }
        for i in 0..k {
            for j in 0..i {
                let mut a = u.clone();
                a[i] += steps[i];
                a[j] += steps[j];
                let fij = post_u(m, &mut work, free, &a);
                let v = (fij - fp[i] - fp[j] + f0) / (steps[i] * steps[j]);
                hess[(i, j)] = v;
                hess[(j, i)] = v;
            }
        }
        if g.iter().any(|v| !v.is_finite()) || (0..k).any(|i| !hess[(i, i)].is_finite()) {
            break;
        }
        // negative Hessian, ridge until positive definite
        let mut nh = hess.scale(-1.0);
        let mut ridge = 0.0;
        let mut chol_ok = nh.cholesky().is_ok();
        while !chol_ok && ridge < 1e8 {
            ridge = if ridge == 0.0 { 1e-6 } else { ridge * 10.0 };
            let mut a = hess.scale(-1.0);
            for i in 0..k {
                a[(i, i)] += ridge * (1.0 + hess[(i, i)].abs());
            }
            chol_ok = a.cholesky().is_ok();
            nh = a;
        }
        let Ok(inv) = nh.inverse() else { break };
        let step: Vec<f64> = (0..k).map(|i| (0..k).map(|j| inv[(i, j)] * g[j]).sum()).collect();
        if ridge == 0.0 {
            hinv = Some(inv.clone());
        }
        let gnorm: f64 = step.iter().zip(&g).map(|(a, b)| a * b).sum::<f64>().abs();
        let mut t = 1.0;
        let mut moved = false;
        for _ in 0..30 {
            // stay within a bounded distance of the start (flat directions)
            let cand: Vec<f64> = u.iter().zip(&step).zip(&u_start).map(|((a, b), s0)| (a + t * b.clamp(-5.0, 5.0)).clamp(s0 - 20.0, s0 + 20.0)).collect();
            let fc = post_u(m, &mut work, free, &cand);
            if fc.is_finite() && fc >= f0 - 1e-12 {
                moved = fc > f0 + 1e-10;
                u = cand;
                f0 = fc;
                break;
            }
            t *= 0.5;
        }
        if !moved || gnorm < 1e-9 {
            break;
        }
    }
    for (k2, &j) in free.iter().enumerate() {
        th[j] = to_x(m.params[j].support, u[k2]);
    }
    hinv
}

// ---------------------------------------------------------------------------
// Gibbs detection
// ---------------------------------------------------------------------------

fn uni_prior_of(m: &Model, j: usize) -> Option<(Uni, &Vec<Arg>)> {
    m.priors.iter().find(|p| p.params.contains(&j)).and_then(|p| match &p.kind {
        PK::Uni(u, a) => Some((*u, a)),
        _ => None,
    })
}

/// Is likelihood `li` a plain normal linear model on equation `eq`?
fn normal_linear(m: &Model, li: usize) -> Option<usize> {
    let l = &m.liks[li];
    if let Fam::Normal(_) = l.fam {
        if l.eqs.len() == 1 && m.eqs[l.eqs[0]].nl.is_none() {
            return Some(l.eqs[0]);
        }
    }
    None
}

fn liks_using(m: &Model, j: usize) -> Vec<usize> {
    (0..m.liks.len()).filter(|&i| m.lik_deps[i].binary_search(&j).is_ok()).collect()
}

fn priors_using_arg(m: &Model, j: usize) -> Vec<usize> {
    (0..m.priors.len()).filter(|&i| m.prior_deps[i].binary_search(&j).is_ok() && !m.priors[i].params.contains(&j)).collect()
}

pub fn gibbs_kind(m: &Model, idx: &[usize]) -> Option<Kind> {
    // coefficients of one normal linear equation, normal or flat priors with numeric args
    let liks: Vec<usize> = {
        let mut v: Vec<usize> = idx.iter().flat_map(|&j| liks_using(m, j)).collect();
        v.sort();
        v.dedup();
        v
    };
    if idx.iter().all(|&j| m.params[j].re.is_none() && m.params[j].mat.is_none()) && liks.len() == 1 {
        if let Some(eq) = normal_linear(m, liks[0]) {
            let e = &m.eqs[eq];
            if idx.iter().all(|j| e.coefs.contains(j)) && idx.iter().all(|&j| priors_using_arg(m, j).is_empty()) {
                let mut means = Vec::new();
                let mut vars = Vec::new();
                let mut ok = true;
                for &j in idx {
                    match uni_prior_of(m, j) {
                        Some((Uni::Normal, a)) if a.len() == 2 && a[0].num().is_some() && a[1].num().is_some() => {
                            means.push(a[0].num().unwrap());
                            vars.push(a[1].num().unwrap());
                        }
                        Some((Uni::Flat, _)) | None => {
                            means.push(0.0);
                            vars.push(f64::INFINITY);
                        }
                        _ => ok = false,
                    }
                }
                if ok {
                    return Some(Kind::GibbsCoef(eq, liks[0], means, vars));
                }
            }
        }
    }
    if idx.len() == 1 {
        let j = idx[0];
        let ab = match uni_prior_of(m, j) {
            Some((Uni::IGamma, a)) if a.len() == 2 && a[0].num().is_some() && a[1].num().is_some() => Some((a[0].num().unwrap(), a[1].num().unwrap())),
            Some((Uni::Jeffreys, _)) => Some((0.0, 0.0)),
            _ => None,
        };
        if let Some((a, b)) = ab {
            let users = priors_using_arg(m, j);
            // variance of a normal likelihood
            if liks.len() == 1 && users.is_empty() {
                if let Fam::Normal(Arg::Par(v)) = &m.liks[liks[0]].fam {
                    if *v == j && normal_linear(m, liks[0]).is_some() || (*v == j && m.liks[liks[0]].eqs.len() == 1) {
                        return Some(Kind::GibbsIgLik(liks[0], a, b));
                    }
                }
            }
            // variance of normal priors
            if liks.is_empty() && !users.is_empty() {
                let mut members = Vec::new();
                for &pi in &users {
                    match &m.priors[pi].kind {
                        PK::Uni(Uni::Normal, args) if args.len() == 2 && matches!(args[1], Arg::Par(v) if v == j) => {
                            let mut dm = Vec::new();
                            args[0].params(&mut dm);
                            if dm.contains(&j) {
                                return None;
                            }
                            for &p in &m.priors[pi].params {
                                members.push((p, pi));
                            }
                        }
                        _ => return None,
                    }
                }
                return Some(Kind::GibbsIgPrior(members, a, b));
            }
        }
    }
    // covariance matrix with an inverse-Wishart prior, used by one
    // multivariate normal likelihood or by multivariate normal priors
    if let Some(mi) = m.mats.iter().position(|mp| mp.idx.len() == idx.len() && mp.idx.iter().all(|j| idx.contains(j))) {
        let own: Vec<usize> = (0..m.priors.len()).filter(|&i| m.priors[i].params.iter().any(|j| idx.contains(j))).collect();
        if own.len() == 1 {
            if let PK::IWishart(df, v, k) = &m.priors[own[0]].kind {
                if *k == mi {
                    let users = priors_using_arg(m, idx[0]);
                    let lik_ok = liks.is_empty() || (liks.len() == 1 && matches!(m.liks[liks[0]].fam, Fam::MvNormal(kk) if kk == mi) && m.liks[liks[0]].eqs.iter().all(|&e| m.eqs[e].nl.is_none()));
                    let pri_ok = users.iter().all(|&pi| matches!(&m.priors[pi].kind, PK::MvNormal(_, super::model::Cov::Mat(kk)) if *kk == mi));
                    if lik_ok && pri_ok && (liks.len() + users.len()) > 0 {
                        return Some(Kind::GibbsIw(mi, *df, v.clone()));
                    }
                }
            }
        }
    }
    // all effects of one random-effects term in a normal linear model
    if let Some(t) = m.params[idx[0]].re {
        let term = &m.re_terms[t];
        if idx.len() == term.params.len() && idx.iter().all(|j| term.params.contains(j)) && liks.len() == 1 && normal_linear(m, liks[0]).is_some() {
            if idx.iter().all(|&j| matches!(uni_prior_of(m, j), Some((Uni::Normal, a)) if a.len() == 2 && a[0].num().is_some())) {
                return Some(Kind::GibbsRe(t, liks[0]));
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Gibbs draws
// ---------------------------------------------------------------------------

fn gibbs(m: &Model, kind: &Kind, idx: &[usize], th: &mut [f64], rng: &mut Rng) {
    match kind {
        Kind::GibbsCoef(eq, li, means, vars) => {
            let e = &m.eqs[*eq];
            let s2 = match &m.liks[*li].fam {
                Fam::Normal(a) => a.val(th),
                _ => return,
            };
            if !(s2 > 0.0) {
                return;
            }
            let cols: Vec<usize> = idx.iter().map(|j| e.coefs.iter().position(|c| c == j).unwrap()).collect();
            // residual without the block's contribution
            let xbuf = vec![0.0; m.eqs.len()];
            let resid: Vec<f64> = (0..m.n).map(|r| {
                let mut f = m.xb(*eq, th, r, &xbuf);
                for &c in &cols {
                    f -= e.x[c][r] * th[e.coefs[c]];
                }
                e.y[r] - f
            }).collect();
            let d = cols.len();
            let mut p = Matrix::zeros(d, d);
            let mut rhs = vec![0.0; d];
            for a in 0..d {
                for b in 0..=a {
                    let s: f64 = (0..m.n).map(|r| e.x[cols[a]][r] * e.x[cols[b]][r]).sum::<f64>() / s2;
                    p[(a, b)] = s;
                    p[(b, a)] = s;
                }
                rhs[a] = (0..m.n).map(|r| e.x[cols[a]][r] * resid[r]).sum::<f64>() / s2;
                if vars[a].is_finite() {
                    p[(a, a)] += 1.0 / vars[a];
                    rhs[a] += means[a] / vars[a];
                }
            }
            let Ok(pinv) = p.inverse() else { return };
            let mean: Vec<f64> = (0..d).map(|a| (0..d).map(|b| pinv[(a, b)] * rhs[b]).sum()).collect();
            let Ok(l) = pinv.cholesky() else { return };
            let z: Vec<f64> = (0..d).map(|_| rng.normal()).collect();
            for a in 0..d {
                let v: f64 = (0..=a).map(|b| l[(a, b)] * z[b]).sum();
                th[idx[a]] = mean[a] + v;
            }
        }
        Kind::GibbsIgLik(li, a, b) => {
            let eq = m.liks[*li].eqs[0];
            let e = &m.eqs[eq];
            let xbuf = vec![0.0; m.eqs.len()];
            let ssr: f64 = (0..m.n).map(|r| {
                let f = m.xb(eq, th, r, &xbuf);
                (e.y[r] - f).powi(2)
            }).sum();
            let shape = a + m.n as f64 / 2.0;
            let scale = b + ssr / 2.0;
            th[idx[0]] = scale / rng.gamma(shape);
        }
        Kind::GibbsIgPrior(members, a, b) => {
            let mut ss = 0.0;
            for &(p, pi) in members {
                let mean = match &m.priors[pi].kind {
                    PK::Uni(_, args) => args[0].val(th),
                    _ => 0.0,
                };
                ss += (th[p] - mean).powi(2);
            }
            let shape = a + members.len() as f64 / 2.0;
            let scale = b + ss / 2.0;
            th[idx[0]] = scale / rng.gamma(shape);
        }
        Kind::GibbsRe(t, li) => {
            let term = &m.re_terms[*t];
            let eq = m.liks[*li].eqs[0];
            let e = &m.eqs[eq];
            let s2 = match &m.liks[*li].fam {
                Fam::Normal(a) => a.val(th),
                _ => return,
            };
            let xbuf = vec![0.0; m.eqs.len()];
            for &j in &term.params {
                let (mu0, v0) = match uni_prior_of(m, j) {
                    Some((_, a)) => (a[0].val(th), a[1].val(th)),
                    None => (0.0, 1.0),
                };
                let rows = &m.re_rows[j];
                let mut szz = 0.0;
                let mut szr = 0.0;
                for &r in rows {
                    let z = term.z.as_ref().map_or(1.0, |z| z[r]);
                    let f = m.xb(eq, th, r, &xbuf) - z * th[j];
                    szz += z * z;
                    szr += z * (e.y[r] - f);
                }
                let prec = szz / s2 + 1.0 / v0;
                let mean = (szr / s2 + mu0 / v0) / prec;
                th[j] = mean + rng.normal() / prec.sqrt();
            }
        }
        Kind::GibbsIw(mi, df, v) => {
            let mp = &m.mats[*mi];
            let d = mp.d;
            // scatter of the residual (or effect) vectors
            let mut sc = Matrix::zeros(d, d);
            let mut cnt = 0usize;
            for l in &m.liks {
                if matches!(l.fam, Fam::MvNormal(k) if k == *mi) {
                    let xbuf = vec![0.0; m.eqs.len()];
                    for r in 0..m.n {
                        let e: Vec<f64> = l.eqs.iter().map(|&k| m.eqs[k].y[r] - m.xb(k, th, r, &xbuf)).collect();
                        for a in 0..d {
                            for b in 0..d {
                                sc[(a, b)] += e[a] * e[b];
                            }
                        }
                        cnt += 1;
                    }
                }
            }
            for p in &m.priors {
                if let PK::MvNormal(means, super::model::Cov::Mat(k)) = &p.kind {
                    if *k != *mi {
                        continue;
                    }
                    let x: Vec<f64> = p.params.iter().map(|&j| th[j]).collect();
                    let mean: Vec<f64> = if means.len() == x.len() { means.iter().map(|a| a.val(th)).collect() } else if means.len() == 1 { vec![means[0].val(th); x.len()] } else { vec![0.0; x.len()] };
                    for g in 0..x.len() / d {
                        for a in 0..d {
                            for b in 0..d {
                                sc[(a, b)] += (x[g * d + a] - mean[g * d + a]) * (x[g * d + b] - mean[g * d + b]);
                            }
                        }
                        cnt += 1;
                    }
                }
            }
            let psi = v.add(&sc).unwrap();
            if let Some(s) = draw_iwishart(df + cnt as f64, &psi, rng) {
                let mut k = 0;
                for i in 0..d {
                    for j in 0..=i {
                        th[mp.idx[k]] = s[(i, j)];
                        k += 1;
                    }
                }
            }
        }
        _ => {}
    }
}

/// Inverse-Wishart(df, psi) draw by the Bartlett decomposition of the
/// Wishart(df, psi^-1) draw.
fn draw_iwishart(df: f64, psi: &Matrix, rng: &mut Rng) -> Option<Matrix> {
    let d = psi.rows;
    let l = psi.inverse().ok()?.cholesky().ok()?;
    let mut a = Matrix::zeros(d, d);
    for i in 0..d {
        a[(i, i)] = rng.chi2(df - i as f64).sqrt();
        for j in 0..i {
            a[(i, j)] = rng.normal();
        }
    }
    let la = l.mul(&a).ok()?;
    let w = la.mul(&la.transpose()).ok()?;
    w.inverse().ok()
}

// ---------------------------------------------------------------------------
// blocking
// ---------------------------------------------------------------------------

fn mh_block(idx: Vec<usize>, sd: &[f64], cov: Option<&Matrix>) -> Block {
    let d = idx.len();
    let mut c = Matrix::zeros(d, d);
    let mut ok = false;
    if let Some(cv) = cov {
        if let Ok(l) = cv.cholesky() {
            c = l;
            ok = true;
        }
    }
    if !ok {
        for i in 0..d {
            c[(i, i)] = sd.get(i).copied().unwrap_or(0.5);
        }
    }
    Block { idx, kind: Kind::Mh, chol: c, lambda: 2.38 / (d as f64).sqrt(), acc: 0, tries: 0, win_acc: 0, win_tries: 0, hist: Vec::new(), nadapt: 0 }
}

/// Default blocks: coefficients of each equation together, other
/// parameters alone, random effects per level, matrices as one block.
pub fn make_blocks(m: &Model, set: &Settings, hinv: Option<(&[usize], &Matrix)>) -> Vec<Block> {
    let np = m.params.len();
    let mut assigned = vec![false; np];
    let mut groups: Vec<(Vec<usize>, bool)> = Vec::new();
    let mut simplex: Vec<Vec<usize>> = Vec::new();
    for p in &m.priors {
        if let PK::Dirichlet(_) = p.kind {
            if p.params.len() >= 2 && p.params.iter().all(|&j| !assigned[j]) {
                for &j in &p.params {
                    assigned[j] = true;
                }
                simplex.push(p.params.clone());
            }
        }
    }
    for (idx, g, split) in &set.blocks {
        let idx: Vec<usize> = idx.iter().copied().filter(|&j| !assigned[j]).collect();
        for &j in &idx {
            assigned[j] = true;
        }
        if *split {
            for j in idx {
                groups.push((vec![j], *g));
            }
        } else if !idx.is_empty() {
            groups.push((idx, *g));
        }
    }
    for e in &m.eqs {
        let idx: Vec<usize> = e.coefs.iter().copied().filter(|&j| !assigned[j]).collect();
        let mut seen = std::collections::BTreeSet::new();
        let idx: Vec<usize> = idx.into_iter().filter(|j| seen.insert(*j)).collect();
        for &j in &idx {
            assigned[j] = true;
        }
        if !idx.is_empty() {
            groups.push((idx, set.gibbs_all));
        }
    }
    for mp in &m.mats {
        let idx: Vec<usize> = mp.idx.iter().copied().filter(|&j| !assigned[j]).collect();
        for &j in &idx {
            assigned[j] = true;
        }
        if !idx.is_empty() {
            let auto = matches!(gibbs_kind(m, &idx), Some(Kind::GibbsIw(..)));
            groups.push((idx, set.gibbs_all || auto));
        }
    }
    // random-effects terms: one Gibbs block when conjugate, else one block per level
    for t in &m.re_terms {
        let idx: Vec<usize> = t.params.iter().copied().filter(|&j| !assigned[j]).collect();
        if idx.is_empty() {
            continue;
        }
        for &j in &idx {
            assigned[j] = true;
        }
        if idx.len() == t.params.len() && gibbs_kind(m, &idx).is_some() {
            groups.push((idx, true));
        } else {
            for j in idx {
                groups.push((vec![j], false));
            }
        }
    }
    for j in 0..np {
        if !assigned[j] {
            // RE variances: Gibbs when conjugate
            let auto = m.params[j].name.contains("sigma2") && gibbs_kind(m, &[j]).map_or(false, |k| matches!(k, Kind::GibbsIgPrior(..)));
            groups.push((vec![j], set.gibbs_all || auto));
        }
    }
    let simplex_blocks: Vec<Block> = simplex.into_iter().map(|idx| {
        let d = idx.len() - 1;
        let mut b = mh_block(idx, &vec![1.0; d], None);
        b.kind = Kind::Simplex;
        b.lambda = 1.0 / (d as f64).sqrt();
        b
    }).collect();
    let lookup = |j: usize| hinv.and_then(|(free, h)| free.iter().position(|&f| f == j).map(|k| (k, h)));
    groups.into_iter().map(|(idx, g)| {
        if g {
            if let Some(kind) = gibbs_kind(m, &idx) {
                let mut b = mh_block(idx, &[], None);
                b.kind = kind;
                return b;
            }
        }
        if idx.len() == 1 && matches!(m.params[idx[0]].support, Support::Discrete(..)) {
            let mut b = mh_block(idx, &[1.0], None);
            b.kind = Kind::Discrete;
            return b;
        }
        let d = idx.len();
        let all_known = idx.iter().all(|&j| lookup(j).is_some());
        let cov = if all_known {
            let mut c = Matrix::zeros(d, d);
            for a in 0..d {
                for b in 0..d {
                    let (ka, h) = lookup(idx[a]).unwrap();
                    let (kb, _) = lookup(idx[b]).unwrap();
                    c[(a, b)] = h[(ka, kb)];
                }
            }
            Some(c)
        } else {
            None
        };
        let sd: Vec<f64> = idx.iter().map(|&j| {
            if let Some((k, h)) = lookup(j) {
                h[(k, k)].max(1e-12).sqrt()
            } else if m.params[j].re.is_some() {
                0.5
            } else if m.params[j].mat.is_some() {
                0.1
            } else {
                0.5
            }
        }).collect();
        mh_block(idx, &sd, cov.as_ref())
    }).chain(simplex_blocks).collect()
}

// ---------------------------------------------------------------------------
// sampler
// ---------------------------------------------------------------------------

fn mh_step(m: &Model, b: &mut Block, th: &mut Vec<f64>, rng: &mut Rng) {
    let d = b.idx.len();
    let cur = m.local_post(&b.idx, th);
    let old: Vec<f64> = b.idx.iter().map(|&j| th[j]).collect();
    if b.kind == Kind::Simplex {
        let k = b.idx.len();
        let x: Vec<f64> = old.clone();
        let last = x[k - 1].max(1e-300);
        let v: Vec<f64> = (0..k - 1).map(|i| (x[i].max(1e-300) / last).ln()).collect();
        let v1: Vec<f64> = v.iter().map(|vi| vi + b.lambda * rng.normal()).collect();
        let mx = v1.iter().copied().fold(0.0f64, f64::max);
        let den: f64 = (-mx).exp() + v1.iter().map(|vi| (vi - mx).exp()).sum::<f64>();
        let mut x1: Vec<f64> = v1.iter().map(|vi| (vi - mx).exp() / den).collect();
        x1.push((-mx).exp() / den);
        for (t, &j) in b.idx.iter().enumerate() {
            th[j] = x1[t];
        }
        let new = m.local_post(&b.idx, th);
        let jac0: f64 = x.iter().map(|v| v.max(1e-300).ln()).sum();
        let jac1: f64 = x1.iter().map(|v| v.max(1e-300).ln()).sum();
        b.tries += 1;
        b.win_tries += 1;
        let ratio = new + jac1 - cur - jac0;
        if new.is_finite() && (ratio >= 0.0 || rng.uniform().ln() < ratio) {
            b.acc += 1;
            b.win_acc += 1;
        } else {
            for (t, &j) in b.idx.iter().enumerate() {
                th[j] = old[t];
            }
        }
        return;
    }
    if b.kind == Kind::Discrete {
        let j = b.idx[0];
        let (lo, hi) = match m.params[j].support {
            Support::Discrete(lo, hi) => (lo, hi),
            _ => (f64::NEG_INFINITY, f64::INFINITY),
        };
        let step = if hi - lo == 1.0 { if th[j] == lo { 1.0 } else { -1.0 } } else if rng.uniform() < 0.5 { 1.0 } else { -1.0 };
        let cand = th[j] + step;
        b.tries += 1;
        b.win_tries += 1;
        if cand < lo || cand > hi {
            return;
        }
        th[j] = cand;
        let new = m.local_post(&b.idx, th);
        if new.is_finite() && (new - cur >= 0.0 || rng.uniform().ln() < new - cur) {
            b.acc += 1;
            b.win_acc += 1;
        } else {
            th[j] = old[0];
        }
        return;
    }
    let u0: Vec<f64> = b.idx.iter().map(|&j| to_u(m.params[j].support, th[j])).collect();
    let jac0: f64 = b.idx.iter().zip(&u0).map(|(&j, &u)| log_jac(m.params[j].support, u)).sum();
    let z: Vec<f64> = (0..d).map(|_| rng.normal()).collect();
    let mut u1 = u0.clone();
    for a in 0..d {
        let mut v = 0.0;
        for c in 0..=a {
            v += b.chol[(a, c)] * z[c];
        }
        u1[a] += b.lambda * v;
    }
    for (k, &j) in b.idx.iter().enumerate() {
        th[j] = to_x(m.params[j].support, u1[k]);
    }
    let jac1: f64 = b.idx.iter().zip(&u1).map(|(&j, &u)| log_jac(m.params[j].support, u)).sum();
    let new = m.local_post(&b.idx, th);
    b.tries += 1;
    b.win_tries += 1;
    let ratio = new + jac1 - cur - jac0;
    if new.is_finite() && (ratio >= 0.0 || rng.uniform().ln() < ratio) {
        b.acc += 1;
        b.win_acc += 1;
    } else {
        for (k, &j) in b.idx.iter().enumerate() {
            th[j] = old[k];
        }
    }
}

fn adapt(m: &Model, b: &mut Block, set: &Settings) {
    if !matches!(b.kind, Kind::Mh | Kind::Simplex) || b.win_tries == 0 || b.nadapt >= set.maxiter {
        return;
    }
    let d = b.idx.len();
    let r = b.win_acc as f64 / b.win_tries as f64;
    let tar = target(d, set.tarate);
    b.nadapt += 1;
    let gamma = 1.0 / (b.nadapt as f64).powf(0.5);
    b.lambda *= ((r - tar) * 2.0 * gamma.max(0.1)).exp();
    b.lambda = b.lambda.clamp(1e-4, 50.0);
    // empirical covariance of the burn-in history
    let n = b.hist.len();
    if n >= 20 * d + 100 {
        let start = n / 2;
        let h = &b.hist[start..];
        let nn = h.len() as f64;
        let mean: Vec<f64> = (0..d).map(|a| h.iter().map(|x| x[a]).sum::<f64>() / nn).collect();
        let mut c = Matrix::zeros(d, d);
        for x in h {
            for a in 0..d {
                for c2 in 0..=a {
                    c[(a, c2)] += (x[a] - mean[a]) * (x[c2] - mean[c2]) / (nn - 1.0);
                }
            }
        }
        for a in 0..d {
            for c2 in 0..a {
                c[(c2, a)] = c[(a, c2)];
            }
        }
        let tr: f64 = (0..d).map(|a| c[(a, a)]).sum::<f64>() / d as f64;
        if tr > 0.0 && tr.is_finite() {
            for a in 0..d {
                c[(a, a)] += 1e-8 * tr.max(1e-12) + 1e-12;
            }
            if let Ok(l) = c.cholesky() {
                // rescale lambda so the step is comparable to the previous one
                b.chol = l;
                b.lambda = b.lambda.max(0.5 / (d as f64).sqrt()).min(2.38 * 1.5 / (d as f64).sqrt());
            }
        }
    }
    let _ = m;
    b.win_acc = 0;
    b.win_tries = 0;
}

/// Run the sampler from `init` (original scale).
pub fn run(m: &Model, init: &[f64], set: &Settings, rng: &mut Rng, hinv: Option<(&[usize], &Matrix)>) -> Result<Sim, String> {
    let np = m.params.len();
    let mut draws = Vec::new();
    let mut lls = Vec::new();
    let mut lps = Vec::new();
    let mut chain = Vec::new();
    let mut arates = Vec::new();
    let mut last_blocks = Vec::new();
    let mut any_gibbs = false;
    let mut any_mh = false;
    for c in 0..set.nchains.max(1) {
        let mut th = init.to_vec();
        let mut blocks = make_blocks(m, set, hinv);
        if c > 0 {
            // overdispersed start: jitter on the unconstrained scale
            for b in &blocks {
                if b.kind != Kind::Mh {
                    continue;
                }
                let d = b.idx.len();
                let z: Vec<f64> = (0..d).map(|_| rng.normal()).collect();
                let save: Vec<f64> = b.idx.iter().map(|&j| th[j]).collect();
                for a in 0..d {
                    let j = b.idx[a];
                    let v: f64 = (0..=a).map(|k| b.chol[(a, k)] * z[k]).sum();
                    let s = m.params[j].support;
                    th[j] = to_x(s, to_u(s, th[j]) + 2.0 * v);
                }
                if !m.log_post(&th).is_finite() {
                    for (k, &j) in b.idx.iter().enumerate() {
                        th[j] = save[k];
                    }
                }
            }
        }
        if !m.log_post(&th).is_finite() {
            return Err(crate::eval::rc_err(498, "initial values not feasible; specify initial() or check the priors"));
        }
        let total = set.burnin + set.mcmcsize * set.thin.max(1);
        let mut kept = 0usize;
        for it in 0..total {
            let burn = it < set.burnin;
            for b in blocks.iter_mut() {
                match b.kind {
                    Kind::Mh | Kind::Discrete | Kind::Simplex => mh_step(m, b, &mut th, rng),
                    _ => {
                        let k = b.kind.clone();
                        gibbs(m, &k, &b.idx.clone(), &mut th, rng);
                        b.acc += 1;
                        b.tries += 1;
                    }
                }
                if burn && b.kind == Kind::Mh {
                    let u: Vec<f64> = b.idx.iter().map(|&j| to_u(m.params[j].support, th[j])).collect();
                    if b.hist.len() < 20000 {
                        b.hist.push(u);
                    }
                }
            }
            if burn && (it + 1) % set.every.max(1) == 0 {
                for b in blocks.iter_mut() {
                    adapt(m, b, set);
                }
            }
            if it + 1 == set.burnin {
                for b in blocks.iter_mut() {
                    b.acc = 0;
                    b.tries = 0;
                    b.hist = Vec::new();
                }
            }
            if !burn && (it - set.burnin + 1) % set.thin.max(1) == 0 {
                kept += 1;
                lls.push(m.loglik(&th));
                lps.push(m.log_prior(&th));
                draws.push(th.clone());
                chain.push(c);
            }
        }
        let _ = kept;
        // overall acceptance rate: average over MH blocks (Gibbs blocks accept always)
        let mh: Vec<&Block> = blocks.iter().filter(|b| matches!(b.kind, Kind::Mh | Kind::Discrete | Kind::Simplex)).collect();
        any_gibbs |= blocks.iter().any(|b| !matches!(b.kind, Kind::Mh | Kind::Discrete | Kind::Simplex));
        any_mh |= !mh.is_empty();
        let ar = if mh.is_empty() { 1.0 } else { mh.iter().map(|b| b.acc as f64 / b.tries.max(1) as f64).sum::<f64>() / mh.len() as f64 };
        arates.push(ar);
        last_blocks = blocks;
    }
    let _ = np;
    Ok(Sim { draws, ll: lls, lp: lps, chain, arate: arates, blocks: last_blocks, any_gibbs, any_mh, start: init.to_vec() })
}
