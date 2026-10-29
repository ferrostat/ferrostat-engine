//! `mi impute`: univariate imputation (regress, pmm, logit, ologit,
//! mlogit), multivariate normal data augmentation (mvn) and chained
//! equations (chained).  Imputations are proper: parameters are drawn from
//! their posterior (or its large-sample normal approximation) before the
//! missing values are drawn.  All draws come from the session generator,
//! so `set seed` or `rseed()` makes results reproducible.

use ferrostat_cmd::eval::rc_err;
use ferrostat_cmd::matrix::Matrix;
use ferrostat_cmd::rng::Rng;
use ferrostat_cmd::svy_est::fit_ml;
use ferrostat_cmd::{dist, util, varlist};
use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::parser::{find_top_level, parse_expression, split_options};

use crate::interpreter::AdoInterpreter;
use crate::mi::{save, MiData};

type Opts = Vec<(String, Option<String>)>;

fn has(opts: &Opts, name: &str, min: usize) -> bool {
    opts.iter().any(|(o, _)| o.len() >= min && name.starts_with(o.to_lowercase().as_str()))
}

fn arg(opts: &Opts, name: &str, min: usize) -> Option<String> {
    opts.iter().find(|(o, _)| o.len() >= min && name.starts_with(o.to_lowercase().as_str())).and_then(|(_, a)| a.clone())
}

fn num(opts: &Opts, name: &str, min: usize) -> Result<Option<f64>, String> {
    match arg(opts, name, min) {
        Some(a) => a.trim().parse().map(Some).map_err(|_| rc_err(198, format!("option {name}() invalid"))),
        None => Ok(None),
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Method {
    Regress,
    Pmm,
    Logit,
    Ologit,
    Mlogit,
    Poisson,
}

fn method_of(s: &str) -> Result<Method, String> {
    let s = s.trim().to_lowercase();
    Ok(match s.as_str() {
        x if x.len() >= 3 && "regress".starts_with(x) => Method::Regress,
        "pmm" => Method::Pmm,
        "logit" | "logistic" => Method::Logit,
        "ologit" => Method::Ologit,
        "mlogit" => Method::Mlogit,
        "poisson" => Method::Poisson,
        _ => return Err(rc_err(198, format!("mi impute: method {s} not supported"))),
    })
}

fn method_title(m: Method) -> &'static str {
    match m {
        Method::Regress => "Linear regression",
        Method::Pmm => "Predictive mean matching",
        Method::Logit => "Logistic regression",
        Method::Ologit => "Ordered logistic regression",
        Method::Mlogit => "Multinomial logistic regression",
        Method::Poisson => "Poisson regression",
    }
}

/// One variable to impute with its conditional model.
#[derive(Clone, Debug)]
struct Target {
    var: String,
    method: Method,
    knn: usize,
}

// ---------------------------------------------------------------------------
// Linear algebra and draws
// ---------------------------------------------------------------------------

fn chol_lower(a: &Matrix) -> Matrix {
    let n = a.rows;
    let mut l = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[(i, j)];
            for k in 0..j {
                s -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                l[(i, j)] = s.max(0.0).sqrt();
            } else {
                l[(i, j)] = if l[(j, j)] > 0.0 { s / l[(j, j)] } else { 0.0 };
            }
        }
    }
    l
}

/// b + chol(V) z
fn mvn_draw(rng: &mut Rng, b: &[f64], v: &Matrix) -> Vec<f64> {
    let l = chol_lower(v);
    let z: Vec<f64> = (0..b.len()).map(|_| rng.normal()).collect();
    (0..b.len()).map(|i| b[i] + (0..=i).map(|j| l[(i, j)] * z[j]).sum::<f64>()).collect()
}

fn xtx_xty(x: &[Vec<f64>], y: &[f64], rows: &[usize]) -> (Matrix, Vec<f64>) {
    let k = x.len();
    let mut a = Matrix::zeros(k, k);
    let mut b = vec![0.0; k];
    for &t in rows {
        for i in 0..k {
            let xi = x[i][t];
            b[i] += xi * y[t];
            for j in 0..k {
                a[(i, j)] += xi * x[j][t];
            }
        }
    }
    (a, b)
}

fn xb(x: &[Vec<f64>], b: &[f64], t: usize) -> f64 {
    x.iter().zip(b).map(|(c, bj)| c[t] * bj).sum()
}

/// Impute `y` at rows `mis` from the model fitted on rows `obs`.
fn impute_one(rng: &mut Rng, method: Method, knn: usize, x: &[Vec<f64>], xcons: bool, y: &[f64], obs: &[usize], mis: &[usize]) -> Result<Vec<f64>, String> {
    if obs.is_empty() {
        return Err(rc_err(2000, "no complete observations to fit the imputation model"));
    }
    match method {
        Method::Regress | Method::Pmm => {
            let (a, c) = xtx_xty(x, y, obs);
            let (ai, dropped) = a.sym_inverse(1e-13);
            let k = x.len() - dropped.len();
            let bh: Vec<f64> = (0..x.len()).map(|i| (0..x.len()).map(|j| ai[(i, j)] * c[j]).sum()).collect();
            let rss: f64 = obs.iter().map(|&t| (y[t] - xb(x, &bh, t)).powi(2)).sum();
            let df = obs.len() as f64 - k as f64;
            if df < 1.0 {
                return Err(rc_err(2001, "insufficient observations to fit the imputation model"));
            }
            let sig2 = rss / rng.chi2(df);
            let bs = mvn_draw(rng, &bh, &ai.scale(sig2));
            if method == Method::Regress {
                let s = sig2.sqrt();
                return Ok(mis.iter().map(|&t| xb(x, &bs, t) + s * rng.normal()).collect());
            }
            // predictive mean matching
            let yo: Vec<(f64, f64)> = obs.iter().map(|&t| (xb(x, &bh, t), y[t])).collect();
            let mut out = Vec::new();
            for &t in mis {
                let p = xb(x, &bs, t);
                let mut d: Vec<(f64, f64)> = yo.iter().map(|(f, yy)| ((f - p).abs(), *yy)).collect();
                d.sort_by(|a, b| a.0.total_cmp(&b.0));
                let kk = knn.max(1).min(d.len());
                let pick = ((rng.uniform() * kk as f64) as usize).min(kk - 1);
                out.push(d[pick].1);
            }
            Ok(out)
        }
        Method::Logit => {
            let mut lv: Vec<f64> = obs.iter().map(|&t| y[t]).collect();
            lv.sort_by(f64::total_cmp);
            lv.dedup();
            if lv.len() != 2 {
                return Err(rc_err(450, "logit imputation requires a binary variable"));
            }
            let xo: Vec<Vec<f64>> = x.iter().map(|c| obs.iter().map(|&t| c[t]).collect()).collect();
            let yo: Vec<f64> = obs.iter().map(|&t| if y[t] == lv[1] { 1.0 } else { 0.0 }).collect();
            let (b, v) = fit_ml("logit", &xo, &yo, &[0.0, 1.0], 0)?;
            let bs = mvn_draw(rng, &b, &v);
            Ok(mis.iter().map(|&t| if rng.uniform() < dist::invlogit(xb(x, &bs, t)) { lv[1] } else { lv[0] }).collect())
        }
        Method::Poisson => {
            let xo: Vec<Vec<f64>> = x.iter().map(|c| obs.iter().map(|&t| c[t]).collect()).collect();
            let yo: Vec<f64> = obs.iter().map(|&t| y[t]).collect();
            if yo.iter().any(|&v| v < 0.0 || v.fract() != 0.0) {
                return Err(rc_err(459, "poisson imputation requires nonnegative integer values"));
            }
            let (b, v) = fit_ml("poisson", &xo, &yo, &[], 0)?;
            let bs = mvn_draw(rng, &b, &v);
            Ok(mis.iter().map(|&t| rng.poisson(xb(x, &bs, t).exp())).collect())
        }
        Method::Ologit | Method::Mlogit => {
            let mut cats: Vec<f64> = obs.iter().map(|&t| y[t]).collect();
            cats.sort_by(f64::total_cmp);
            cats.dedup();
            if cats.len() < 2 {
                return Ok(mis.iter().map(|_| cats[0]).collect());
            }
            let nc = cats.len();
            // ordered models have no constant
            let xs: Vec<Vec<f64>> = if method == Method::Ologit && xcons { x[..x.len() - 1].to_vec() } else { x.to_vec() };
            let xo: Vec<Vec<f64>> = xs.iter().map(|c| obs.iter().map(|&t| c[t]).collect()).collect();
            let yo: Vec<f64> = obs.iter().map(|&t| y[t]).collect();
            let base = (0..nc).max_by_key(|&c| (yo.iter().filter(|&&v| v == cats[c]).count(), std::cmp::Reverse(c))).unwrap();
            let fam = if method == Method::Ologit { "ologit" } else { "mlogit" };
            let (b, v) = fit_ml(fam, &xo, &yo, &cats, base)?;
            let bs = mvn_draw(rng, &b, &v);
            let kx = xs.len();
            let mut out = Vec::new();
            for &t in mis {
                let probs: Vec<f64> = if method == Method::Ologit {
                    let e = xb(&xs, &bs[..kx], t);
                    let mut cut: Vec<f64> = bs[kx..].to_vec();
                    cut.sort_by(f64::total_cmp);
                    let f = |c: usize| if c == nc - 1 { 1.0 } else { dist::invlogit(cut[c] - e) };
                    (0..nc).map(|c| f(c) - if c == 0 { 0.0 } else { f(c - 1) }).collect()
                } else {
                    let mut eta = vec![0.0; nc];
                    for (eq, c) in (0..nc).filter(|&c| c != base).enumerate() {
                        eta[c] = xb(&xs, &bs[eq * kx..(eq + 1) * kx], t);
                    }
                    let m = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let ex: Vec<f64> = eta.iter().map(|e| (e - m).exp()).collect();
                    let s: f64 = ex.iter().sum();
                    ex.iter().map(|e| e / s).collect()
                };
                let u = rng.uniform();
                let mut acc = 0.0;
                let mut pick = nc - 1;
                for (c, p) in probs.iter().enumerate() {
                    acc += p;
                    if u < acc {
                        pick = c;
                        break;
                    }
                }
                out.push(cats[pick]);
            }
            Ok(out)
        }
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Split `main [if exp]`.
fn split_if(s: &str) -> (String, Option<String>) {
    let t = format!(" {s} ");
    match t.find(" if ") {
        Some(i) => (t[..i].trim().to_string(), Some(t[i + 4..].trim().to_string())),
        None => (s.trim().to_string(), None),
    }
}

/// Predictor columns from a list of variables (with `i.` factor notation).
fn predictors(ds: &DataSet, spec: &str) -> Result<(Vec<String>, Vec<(String, Option<f64>)>), String> {
    let mut vars = Vec::new();
    let mut cols = Vec::new();
    for tok in spec.split_whitespace() {
        if let Some(v) = tok.strip_prefix("i.") {
            for name in varlist::expand(ds, v)? {
                vars.push(name.clone());
                let c = ds.get_var(&name).unwrap();
                let mut lv: Vec<f64> = (0..ds.n_obs).map(|i| c.num(i)).filter(|x| !is_missing(*x)).collect();
                lv.sort_by(f64::total_cmp);
                lv.dedup();
                for l in lv.into_iter().skip(1) {
                    cols.push((name.clone(), Some(l)));
                }
            }
        } else {
            for name in varlist::expand(ds, tok.trim_start_matches("c."))? {
                vars.push(name.clone());
                cols.push((name, None));
            }
        }
    }
    Ok((vars, cols))
}

fn column(ds: &DataSet, var: &str, level: Option<f64>, vals: Option<&Vec<f64>>) -> Vec<f64> {
    let base: Vec<f64> = match vals {
        Some(v) => v.clone(),
        None => (0..ds.n_obs).map(|i| ds.get_var(var).map_or(SYSMISS, |c| c.num(i))).collect(),
    };
    match level {
        Some(l) => base.iter().map(|&x| if is_missing(x) { SYSMISS } else if x == l { 1.0 } else { 0.0 }).collect(),
        None => base,
    }
}

// ---------------------------------------------------------------------------
// mi impute
// ---------------------------------------------------------------------------

pub(crate) fn impute(me: &mut AdoInterpreter, args: &str) -> Result<(), String> {
    let a = args.trim();
    let (meth, rest) = a.split_once(char::is_whitespace).ok_or_else(|| rc_err(198, "mi impute: method required"))?;
    let meth = meth.to_lowercase();
    if matches!(meth.as_str(), "intreg" | "nbreg" | "truncreg") { return crate::mi_more::impute_extra(me, &meth, rest); }
    let (main, opts): (String, Opts) = match find_top_level_outside_parens(rest) {
        Some(i) => (rest[..i].to_string(), split_options(&rest[i + 1..])),
        None => (rest.to_string(), Vec::new()),
    };
    if !MiData::is_set(&me.engine.dataset) {
        return Err(rc_err(119, "no; data not mi set"));
    }
    let mut md = MiData::load(&me.engine.dataset)?;
    let add = num(&opts, "add", 3)?.map(|x| x as usize);
    let replace = has(&opts, "replace", 7);
    if add.is_none() && !replace {
        return Err(rc_err(198, "either add() or replace is required"));
    }
    if let Some(s) = arg(&opts, "rseed", 5) {
        let seed: u64 = s.trim().parse().map_err(|_| rc_err(198, "rseed() invalid"))?;
        me.engine.store.rng.borrow_mut().reseed(seed);
    }
    let first_new = if replace && add.is_none() { 1 } else { md.m + 1 };
    let new_m = md.m + add.unwrap_or(0);
    let ims: Vec<usize> = if replace && add.is_none() { (1..=md.m).collect() } else { (first_new..=new_m).collect() };
    if ims.is_empty() {
        return Err(rc_err(198, "no imputations to fill; specify add(#)"));
    }
    let base = md.base.clone();
    let n = base.n_obs;
    let knn_default = num(&opts, "knn", 3)?.map(|x| x as usize).unwrap_or(1);
    let noconstant = has(&opts, "noconstant", 3);

    // targets and predictors
    let (main, cond) = split_if(&main);
    let mut targets: Vec<Target> = Vec::new();
    let pred_spec: String;
    let mut chained_burnin = 10usize;
    let mvn = meth == "mvn";
    let monotone = meth == "monotone";
    let chained = meth == "chained" || monotone;
    if chained {
        chained_burnin = num(&opts, "burnin", 5)?.map(|x| x as usize).unwrap_or(10);
        let (lhs, rhs) = match find_top_level_outside_parens_char(&main, '=') {
            Some(i) => (main[..i].to_string(), main[i + 1..].to_string()),
            None => (main.clone(), String::new()),
        };
        pred_spec = rhs;
        let mut s = lhs.trim().to_string();
        while let Some(stripped) = s.strip_prefix('(') {
            let mut depth = 1i32;
            let mut close = None;
            for (i, c) in stripped.char_indices() {
                match c {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            close = Some(i);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let close = close.ok_or_else(|| rc_err(198, "unmatched parenthesis"))?;
            let inside = &stripped[..close];
            let (m, sub) = match find_top_level(inside, ',') {
                Some(i) => (inside[..i].to_string(), split_options(&inside[i + 1..])),
                None => (inside.to_string(), Vec::new()),
            };
            let method = method_of(m.split_whitespace().next().unwrap_or(""))?;
            let knn = num(&sub, "knn", 3)?.map(|x| x as usize).unwrap_or(knn_default);
            let after = stripped[close + 1..].trim_start();
            let end = after.find('(').unwrap_or(after.len());
            for v in varlist::expand(&base, &after[..end])? {
                targets.push(Target { var: v, method, knn });
            }
            s = after[end..].trim().to_string();
        }
        if !s.is_empty() {
            return Err(rc_err(198, format!("mi impute chained: invalid specification {s}")));
        }
    } else if mvn {
        let (lhs, rhs) = main.split_once('=').map(|(a, b)| (a.to_string(), b.to_string())).unwrap_or((main.clone(), String::new()));
        pred_spec = rhs;
        for v in varlist::expand(&base, &lhs)? {
            targets.push(Target { var: v, method: Method::Regress, knn: 1 });
        }
    } else {
        let method = method_of(&meth)?;
        let mut w = main.split_whitespace();
        let y = w.next().ok_or_else(|| rc_err(100, "varlist required"))?;
        let y = varlist::resolve_name(&base, y)?;
        pred_spec = main.trim()[main.trim().find(char::is_whitespace).unwrap_or(main.trim().len())..].trim().trim_start_matches('=').to_string();
        targets.push(Target { var: y, method, knn: knn_default });
    }
    if targets.is_empty() {
        return Err(rc_err(100, "variables to impute required"));
    }
    for t in &targets {
        if !md.ivars.contains(&t.var) {
            return Err(rc_err(198, format!("variable {} not registered as imputed; use mi register imputed", t.var)));
        }
    }
    let (pvars, pcols) = predictors(&base, &pred_spec)?;
    // estimation sample: if condition and complete predictors
    let mut sample = vec![true; n];
    if let Some(c) = &cond {
        let e = parse_expression(c).map_err(|e| rc_err(198, e.message))?;
        let vals = util::eval_col(&base, &me.engine.store, &e, &vec![true; n])?;
        for i in 0..n {
            sample[i] = !is_missing(vals[i]) && vals[i] != 0.0;
        }
    }
    for v in &pvars {
        if md.ivars.contains(v) {
            continue;
        }
        let c = column(&base, v, None, None);
        for i in 0..n {
            if is_missing(c[i]) {
                sample[i] = false;
            }
        }
    }
    let ycols: Vec<Vec<f64>> = targets.iter().map(|t| column(&base, &t.var, None, None)).collect();
    // only system missing values are filled in
    let to_fill: Vec<Vec<bool>> = ycols.iter().map(|c| (0..n).map(|i| sample[i] && c[i] == SYSMISS).collect()).collect();
    let observed: Vec<Vec<bool>> = ycols.iter().map(|c| (0..n).map(|i| sample[i] && !is_missing(c[i])).collect()).collect();

    // make room for the new imputations
    while md.imp.len() < new_m {
        let mut slot = std::collections::BTreeMap::new();
        for v in md.ivars.iter().chain(&md.pvars) {
            slot.insert(v.clone(), column(&base, v, None, None));
        }
        md.imp.push(slot);
    }
    md.m = new_m.max(md.m);
    let mut filled = vec![usize::MAX; targets.len()];
    let mut burnin_mvn = 0usize;
    let mut between_mvn = 0usize;
    let mut mvn_iters = 0usize;
    let mut trace: Vec<(usize, usize, Vec<Vec<f64>>, Matrix)> = Vec::new();
    {
        let mut rng = me.engine.store.rng.borrow_mut();
        let rng: &mut Rng = &mut rng;
        // predictor columns for imputation m (imputed predictors take their imputed values)
        let xcols = |md: &MiData, m: usize| -> Vec<Vec<f64>> {
            let mut x: Vec<Vec<f64>> = pcols.iter().map(|(v, l)| column(&base, v, *l, if md.ivars.contains(v) { md.imp[m - 1].get(v) } else { None })).collect();
            if !noconstant {
                x.push(vec![1.0; n]);
            }
            x
        };
        if mvn {
            burnin_mvn = num(&opts, "burnin", 5)?.map(|x| x as usize).unwrap_or(100);
            between_mvn = num(&opts, "burnbetween", 6)?.map(|x| x as usize).unwrap_or(100);
            let x = xcols(&md, ims[0]);
            let rows: Vec<usize> = (0..n).filter(|&i| sample[i]).collect();
            let mut y = ycols.clone();
            // start: observed means
            for (j, c) in y.iter_mut().enumerate() {
                let obs: Vec<f64> = rows.iter().filter(|&&i| observed[j][i]).map(|&i| c[i]).collect();
                let m = util::mean(&obs);
                for &i in &rows {
                    if !observed[j][i] {
                        c[i] = m;
                    }
                }
            }
            let (mut beta, mut sigma) = mvn_pstep(rng, &x, &y, &rows, false);
            for (idx, &m) in ims.iter().enumerate() {
                let iters = if idx == 0 { burnin_mvn } else { between_mvn };
                for _ in 0..iters.max(1) {
                    mvn_istep(rng, &x, &mut y, &rows, &observed, &beta, &sigma);
                    let (b2, s2) = mvn_pstep(rng, &x, &y, &rows, true);
                    beta = b2;
                    sigma = s2;
                    mvn_iters += 1;
                    trace.push((m, mvn_iters, beta.clone(), sigma.clone()));
                }
                // imputations from the current parameters
                let mut yi = y.clone();
                mvn_istep(rng, &x, &mut yi, &rows, &observed, &beta, &sigma);
                for (j, t) in targets.iter().enumerate() {
                    let col = md.imp[m - 1].get_mut(&t.var).unwrap();
                    let mut cnt = 0;
                    for i in 0..n {
                        if to_fill[j][i] {
                            col[i] = yi[j][i];
                            cnt += 1;
                        } else {
                            col[i] = ycols[j][i];
                        }
                    }
                    filled[j] = filled[j].min(cnt);
                }
            }
        } else if chained {
            // order: increasing number of missing values unless orderasis
            let mut order: Vec<usize> = (0..targets.len()).collect();
            if !has(&opts, "orderasis", 5) {
                order.sort_by_key(|&j| to_fill[j].iter().filter(|&&b| b).count());
            }
            for &m in &ims {
                let xbase = xcols(&md, m);
                let mut cur = ycols.clone();
                // initial values: draws from the observed values
                for j in 0..targets.len() {
                    if monotone {
                        break;
                    }
                    let pool: Vec<f64> = (0..n).filter(|&i| observed[j][i]).map(|i| ycols[j][i]).collect();
                    if pool.is_empty() {
                        return Err(rc_err(2000, format!("{}: no observed values", targets[j].var)));
                    }
                    for i in 0..n {
                        if to_fill[j][i] {
                            cur[j][i] = pool[((rng.uniform() * pool.len() as f64) as usize).min(pool.len() - 1)];
                        }
                    }
                }
                for _ in 0..if monotone { 1 } else { chained_burnin.max(1) } {
                    for (pos, &j) in order.iter().enumerate() {
                        let mut x: Vec<Vec<f64>> = Vec::new();
                        for (o, t) in targets.iter().enumerate() {
                            if o == j || (monotone && !order[..pos].contains(&o)) {
                                continue;
                            }
                            if matches!(t.method, Method::Ologit | Method::Mlogit) {
                                let mut lv: Vec<f64> = (0..n).filter(|&i| sample[i]).map(|i| cur[o][i]).collect();
                                lv.sort_by(f64::total_cmp);
                                lv.dedup();
                                for l in lv.into_iter().skip(1) {
                                    x.push(cur[o].iter().map(|&v| if v == l { 1.0 } else { 0.0 }).collect());
                                }
                            } else {
                                x.push(cur[o].clone());
                            }
                        }
                        x.extend(xbase.iter().cloned());
                        let obs: Vec<usize> = (0..n).filter(|&i| observed[j][i]).collect();
                        let mis: Vec<usize> = (0..n).filter(|&i| to_fill[j][i]).collect();
                        let vals = impute_one(rng, targets[j].method, targets[j].knn, &x, !noconstant, &cur[j], &obs, &mis)?;
                        for (k, &i) in mis.iter().enumerate() {
                            cur[j][i] = vals[k];
                        }
                    }
                }
                for (j, t) in targets.iter().enumerate() {
                    let col = md.imp[m - 1].get_mut(&t.var).unwrap();
                    *col = cur[j].clone();
                    filled[j] = filled[j].min(to_fill[j].iter().filter(|&&b| b).count());
                }
            }
        } else {
            let t = &targets[0];
            for &m in &ims {
                let x = xcols(&md, m);
                let obs: Vec<usize> = (0..n).filter(|&i| observed[0][i]).collect();
                let mis: Vec<usize> = (0..n).filter(|&i| to_fill[0][i]).collect();
                let vals = impute_one(rng, t.method, t.knn, &x, !noconstant, &ycols[0], &obs, &mis)?;
                let col = md.imp[m - 1].get_mut(&t.var).unwrap();
                *col = ycols[0].clone();
                for (k, &i) in mis.iter().enumerate() {
                    col[i] = vals[k];
                }
                filled[0] = filled[0].min(mis.len());
            }
        }
    }
    save(me, &md)?;
    if let Some(f) = arg(&opts, "savetrace", 5) {
        let xn: Vec<String> = pcols.iter().map(|(v, l)| match l { Some(l) => format!("{}.{v}", ferrostat_cmd::format::fmt_g(*l, 8)), None => v.clone() }).chain((!noconstant).then(|| "_cons".to_string())).collect();
        crate::mi_more::write_trace(&f, &targets.iter().map(|t| t.var.clone()).collect::<Vec<_>>(), &xn, &trace)?;
    }

    // ---- output
    let added = if replace && add.is_none() { 0 } else { ims.len() };
    let updated = ims.len() - added;
    let mut out = Vec::new();
    if chained && !monotone {
        out.push(String::new());
        out.push("Conditional models:".into());
        for (j, t) in targets.iter().enumerate() {
            let others: Vec<String> = targets.iter().enumerate().filter(|(o, _)| *o != j).map(|(_, x)| if matches!(x.method, Method::Ologit | Method::Mlogit) { format!("i.{}", x.var) } else { x.var.clone() }).chain(pvars.iter().cloned()).collect();
            let mname = match t.method {
                Method::Regress => "regress",
                Method::Pmm => "pmm",
                Method::Logit => "logit",
                Method::Ologit => "ologit",
                Method::Mlogit => "mlogit",
                Method::Poisson => "poisson",
            };
            let extra = if t.method == Method::Pmm { format!(" , knn({})", t.knn) } else { String::new() };
            out.push(format!("{:>14}: {mname} {} {}{extra}", t.var, t.var, others.join(" ")));
        }
        out.push(String::new());
        out.push("Performing chained iterations ...".into());
    }
    out.push(String::new());
    let title1 = if mvn || chained { "Multivariate imputation" } else { "Univariate imputation" };
    let title2 = if mvn {
        "Multivariate normal regression"
    } else if monotone {
        "Monotone method"
    } else if chained {
        "Chained equations"
    } else {
        method_title(targets[0].method)
    };
    let range = if ims.len() == 1 { format!("Imputed: m={}", ims[0]) } else { format!("Imputed: m={} through m={}", ims[0], ims[ims.len() - 1]) };
    out.push(format!("{:<44}{:>11} = {:>8}", title1, "Imputations", new_m));
    out.push(format!("{:<44}{:>11} = {:>8}", title2, "added", added));
    out.push(format!("{:<44}{:>11} = {:>8}", range, "updated", updated));
    if mvn {
        out.push(String::new());
        out.push(format!("{:<44}{:>11} = {:>8}", "Prior: uniform", "Iterations", mvn_iters));
        out.push(format!("{:<44}{:>11} = {:>8}", "", "burn-in", burnin_mvn));
        out.push(format!("{:<44}{:>11} = {:>8}", "", "between", between_mvn));
    }
    if monotone {
        out.push(String::new());
        for t in &targets {
            out.push(format!("{:>14}: {}", t.var, method_title(t.method).to_lowercase()));
        }
    }
    if chained && !monotone {
        out.push(String::new());
        out.push(format!("{:<44}{:>11} = {:>8}", "Initialization: random", "Iterations", chained_burnin * ims.len()));
        out.push(format!("{:<44}{:>11} = {:>8}", "", "burn-in", chained_burnin));
        out.push(String::new());
        for t in &targets {
            out.push(format!("{:>14}: {}", t.var, method_title(t.method).to_lowercase()));
        }
    }
    out.push(String::new());
    out.push("-".repeat(66));
    out.push(format!("{:>19}|{:^46}", "", "Observations per m"));
    out.push(format!("{:>19}|{}", "", "-".repeat(46)));
    out.push(format!("{:>18} |   Complete   Incomplete   Imputed |     Total", "Variable"));
    out.push(format!("{}+{}+{}", "-".repeat(19), "-".repeat(35), "-".repeat(10)));
    for (j, t) in targets.iter().enumerate() {
        let comp = observed[j].iter().filter(|&&b| b).count();
        let inc = (0..n).filter(|&i| sample[i] && is_missing(ycols[j][i])).count();
        out.push(format!("{:>18} | {:>10}   {:>10} {:>9} | {:>9}", ferrostat_cmd::summarize::abbrev(&t.var, 18), comp, inc, if filled[j] == usize::MAX { 0 } else { filled[j] }, comp + inc));
    }
    out.push("-".repeat(66));
    out.push("(Complete + Incomplete = Total; Imputed is the minimum across m".into());
    out.push(" of the number of filled-in observations.)".into());
    me.emit(&out.join("\n"));
    let r = me.engine.store.begin_r();
    r.set("M", new_m as f64);
    r.set("M_add", added as f64);
    r.set("M_update", updated as f64);
    r.set("k_ivars", targets.len() as f64);
    r.set_str("method", meth.clone());
    r.set_str("ivars", targets.iter().map(|t| t.var.clone()).collect::<Vec<_>>().join(" "));
    Ok(())
}

/// Top-level comma outside parentheses (chained equations carry their own
/// options inside parentheses).
fn find_top_level_outside_parens(s: &str) -> Option<usize> {
    find_top_level_outside_parens_char(s, ',')
}

fn find_top_level_outside_parens_char(s: &str, ch: char) -> Option<usize> {
    let mut depth = 0i32;
    let mut quote = false;
    for (i, c) in s.char_indices() {
        match c {
            '"' => quote = !quote,
            '(' if !quote => depth += 1,
            ')' if !quote => depth -= 1,
            c if c == ch && depth == 0 && !quote => return Some(i),
            _ => {}
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Multivariate normal data augmentation
// ---------------------------------------------------------------------------

/// P-step: draw (B, Sigma) from the posterior given completed data (uniform
/// prior); with `draw` false the least-squares estimates are returned.
fn mvn_pstep(rng: &mut Rng, x: &[Vec<f64>], y: &[Vec<f64>], rows: &[usize], draw: bool) -> (Vec<Vec<f64>>, Matrix) {
    let q = x.len();
    let p = y.len();
    let (a, _) = xtx_xty(x, &y[0], rows);
    let (ai, _) = a.sym_inverse(1e-13);
    // B-hat: q x p
    let mut bh = vec![vec![0.0; p]; q];
    for j in 0..p {
        let mut xty = vec![0.0; q];
        for &t in rows {
            for i in 0..q {
                xty[i] += x[i][t] * y[j][t];
            }
        }
        for i in 0..q {
            bh[i][j] = (0..q).map(|c| ai[(i, c)] * xty[c]).sum();
        }
    }
    let mut s = Matrix::zeros(p, p);
    for &t in rows {
        let e: Vec<f64> = (0..p).map(|j| y[j][t] - (0..q).map(|i| x[i][t] * bh[i][j]).sum::<f64>()).collect();
        for a1 in 0..p {
            for b1 in 0..p {
                s[(a1, b1)] += e[a1] * e[b1];
            }
        }
    }
    let n = rows.len() as f64;
    if !draw {
        return (bh, s.scale(1.0 / (n - q as f64).max(1.0)));
    }
    // Sigma ~ inverse Wishart(n - q, S): Sigma = W^-1, W ~ Wishart(n - q, S^-1)
    let df = n - q as f64;
    let (si, _) = s.sym_inverse(1e-13);
    let l = chol_lower(&si);
    let mut am = Matrix::zeros(p, p);
    for i in 0..p {
        am[(i, i)] = rng.chi2(df - i as f64).sqrt();
        for j in 0..i {
            am[(i, j)] = rng.normal();
        }
    }
    let la = l.mul(&am).unwrap();
    let w = la.mul(&la.transpose()).unwrap();
    let (sigma, _) = w.sym_inverse(1e-13);
    // B = B-hat + chol((X'X)^-1) Z chol(Sigma)'
    let lx = chol_lower(&ai);
    let ls = chol_lower(&sigma);
    let z: Vec<Vec<f64>> = (0..q).map(|_| (0..p).map(|_| rng.normal()).collect()).collect();
    let mut b = bh.clone();
    for i in 0..q {
        for j in 0..p {
            let mut v = 0.0;
            for k in 0..=i {
                for l2 in 0..=j {
                    v += lx[(i, k)] * z[k][l2] * ls[(j, l2)];
                }
            }
            b[i][j] += v;
        }
    }
    (b, sigma)
}

/// I-step: draw the missing parts of each row from their conditional normal.
fn mvn_istep(rng: &mut Rng, x: &[Vec<f64>], y: &mut [Vec<f64>], rows: &[usize], observed: &[Vec<bool>], b: &[Vec<f64>], sigma: &Matrix) {
    let p = y.len();
    let q = x.len();
    for &t in rows {
        let mis: Vec<usize> = (0..p).filter(|&j| !observed[j][t]).collect();
        if mis.is_empty() {
            continue;
        }
        let obs: Vec<usize> = (0..p).filter(|&j| observed[j][t]).collect();
        let mu: Vec<f64> = (0..p).map(|j| (0..q).map(|i| x[i][t] * b[i][j]).sum()).collect();
        let sub = |r: &[usize], c: &[usize]| {
            let mut m = Matrix::zeros(r.len(), c.len());
            for (a, &i) in r.iter().enumerate() {
                for (d, &j) in c.iter().enumerate() {
                    m[(a, d)] = sigma[(i, j)];
                }
            }
            m
        };
        let (cm, cv) = if obs.is_empty() {
            (mis.iter().map(|&j| mu[j]).collect::<Vec<f64>>(), sub(&mis, &mis))
        } else {
            let soo = sub(&obs, &obs);
            let (soi, _) = soo.sym_inverse(1e-13);
            let smo = sub(&mis, &obs);
            let k = smo.mul(&soi).unwrap();
            let dev: Vec<f64> = obs.iter().map(|&j| y[j][t] - mu[j]).collect();
            let cm: Vec<f64> = mis.iter().enumerate().map(|(a, &j)| mu[j] + (0..obs.len()).map(|d| k[(a, d)] * dev[d]).sum::<f64>()).collect();
            let cv = sub(&mis, &mis).sub(&k.mul(&smo.transpose()).unwrap()).unwrap();
            (cm, cv)
        };
        let draw = mvn_draw(rng, &cm, &cv);
        for (a, &j) in mis.iter().enumerate() {
            y[j][t] = draw[a];
        }
    }
}
