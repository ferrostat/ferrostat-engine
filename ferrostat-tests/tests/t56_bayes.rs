//! Bayesian analysis: bayesmh, the bayes prefix and postestimation.
//!
//! Checks use conjugate models with closed-form posteriors (normal mean with
//! known variance, normal-inverse-gamma regression, beta-binomial,
//! Poisson-gamma), compared with the MCMC estimates within Monte-Carlo
//! error, and the prefix with flat priors against maximum likelihood.

use ferrostat_tests::*;

/// Deterministic data: n observations of x, a normal outcome y, a binary
/// outcome b and a count k.
fn data(s: &mut Session, n: usize) {
    s.script(&format!(
        "set obs {n}
         gen double u = (mod(_n*53, 97) + 0.5)/97
         gen double x = mod(_n*13, 17)/4 - 2
         gen double y = 1 + 0.5*x + invnormal(u)
         gen byte b = (u < invlogit(0.3 + 0.8*x))
         gen double k = floor(-ln(u) * 3)
         gen id = mod(_n, 8) + 1"
    ));
}

fn b(s: &mut Session, name: &str) -> f64 {
    s.scalar(&format!("_b[{name}]"))
}

fn sd(s: &mut Session, name: &str) -> f64 {
    s.scalar(&format!("sqrt(el(e(V), colnumb(e(V), \"{name}\"), colnumb(e(V), \"{name}\")))"))
}

#[test]
fn normal_mean_known_variance() {
    let mut s = Session::new();
    data(&mut s, 50);
    // prior N(0, 100), unit variance: posterior N(v*sum(y), v), v = 1/(n + 1/100)
    s.run("quietly summarize y");
    let (n, sum) = (s.scalar("r(N)"), s.scalar("r(sum)"));
    let v = 1.0 / (n + 0.01);
    let out = s.run("bayesmh y, likelihood(normal(1)) prior({y:_cons}, normal(0,100)) mcmcsize(4000) burnin(500) rseed(11)");
    assert!(out.contains("Bayesian") && out.contains("_cons"), "{out}");
    let m = b(&mut s, "y:_cons");
    assert!((m - v * sum).abs() < 0.03, "{m} vs {}", v * sum);
    let sdv = sd(&mut s, "y:_cons");
    assert!((sdv / v.sqrt() - 1.0).abs() < 0.12, "{sdv} vs {}", v.sqrt());
    // reproducible with the same seed
    s.run("bayesmh y, likelihood(normal(1)) prior({y:_cons}, normal(0,100)) mcmcsize(4000) burnin(500) rseed(11)");
    assert_eq!(b(&mut s, "y:_cons"), m);
}

#[test]
fn beta_binomial_and_poisson_gamma() {
    let mut s = Session::new();
    data(&mut s, 60);
    s.run("quietly summarize b");
    let (n, succ) = (s.scalar("r(N)"), s.scalar("r(sum)"));
    s.run("bayesmh b, likelihood(dbernoulli({p})) prior({p}, beta(2,3)) mcmcsize(4000) burnin(500) rseed(3)");
    let (a1, b1) = (2.0 + succ, 3.0 + n - succ);
    let mean = a1 / (a1 + b1);
    let var = a1 * b1 / ((a1 + b1).powi(2) * (a1 + b1 + 1.0));
    assert!((b(&mut s, "p") - mean).abs() < 0.01, "{} vs {mean}", b(&mut s, "p"));
    assert!((sd(&mut s, "p") / var.sqrt() - 1.0).abs() < 0.12);
    // Poisson-gamma: prior gamma(shape 2, scale 1) -> gamma(2 + sum, scale 1/(1 + n))
    s.run("quietly summarize k");
    let sumk = s.scalar("r(sum)");
    s.run("bayesmh k, likelihood(dpoisson({mu})) prior({mu}, gamma(2,1)) mcmcsize(4000) burnin(500) rseed(5)");
    let mean = (2.0 + sumk) / (1.0 + n);
    let sdv = (2.0 + sumk).sqrt() / (1.0 + n);
    assert!((b(&mut s, "mu") - mean).abs() < 4.0 * sdv / 20.0, "{} vs {mean}", b(&mut s, "mu"));
    assert!((sd(&mut s, "mu") / sdv - 1.0).abs() < 0.12);
}

#[test]
fn normal_inverse_gamma_regression() {
    let mut s = Session::new();
    data(&mut s, 40);
    // conjugate prior beta | s2 ~ N(0, 10 s2 I), s2 ~ IG(2, 1)
    let out = s.run("bayesmh y x, likelihood(normal({s2})) prior({y:}, normal(0, 10*{s2})) prior({s2}, igamma(2,1)) mcmcsize(6000) burnin(1000) rseed(7)");
    assert!(out.contains("Random-walk Metropolis-Hastings"), "{out}");
    // closed form
    let ds = s.ds().clone();
    let get = |n: &str| -> Vec<f64> { let v = ds.get_var(n).unwrap(); v.data.iter().map(|x| x.num()).collect() };
    let (x, y) = (get("x"), get("y"));
    let n = x.len() as f64;
    let (sx, sxx, sy, sxy, syy) = (x.iter().sum::<f64>(), x.iter().map(|v| v * v).sum::<f64>(), y.iter().sum::<f64>(), x.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>(), y.iter().map(|v| v * v).sum::<f64>());
    // Lambda = X'X + I/10 with columns (x, 1)
    let (l11, l12, l22) = (sxx + 0.1, sx, n + 0.1);
    let det = l11 * l22 - l12 * l12;
    let (r1, r2) = (sxy, sy);
    let m1 = (l22 * r1 - l12 * r2) / det;
    let m2 = (l11 * r2 - l12 * r1) / det;
    let an = 2.0 + n / 2.0;
    let bn = 1.0 + 0.5 * (syy - (m1 * r1 + m2 * r2));
    let es2 = bn / (an - 1.0);
    assert!((b(&mut s, "y:x") - m1).abs() < 0.03, "{} vs {m1}", b(&mut s, "y:x"));
    assert!((b(&mut s, "y:_cons") - m2).abs() < 0.04, "{} vs {m2}", b(&mut s, "y:_cons"));
    assert!((b(&mut s, "s2") / es2 - 1.0).abs() < 0.06, "{} vs {es2}", b(&mut s, "s2"));
    // marginal posterior sd of the slope: multivariate t
    let v11 = l22 / det * bn / (an - 1.0);
    assert!((sd(&mut s, "y:x") / v11.sqrt() - 1.0).abs() < 0.15);
}

#[test]
fn bayes_regress_flat_priors_is_ols() {
    let mut s = Session::new();
    data(&mut s, 80);
    s.run("regress y x");
    let (bx, bc, rmse) = (b(&mut s, "x"), b(&mut s, "_cons"), s.scalar("e(rmse)"));
    let sex = s.scalar("_se[x]");
    // Gibbs with flat priors and Jeffreys prior on the variance
    let out = s.run("bayes, prior({y:}, flat) prior({sigma2}, jeffreys) gibbs mcmcsize(3000) burnin(200) rseed(1): regress y x");
    assert!(out.contains("Gibbs sampling") && out.contains("sigma2"), "{out}");
    assert!((b(&mut s, "y:x") - bx).abs() < 0.2 * sex, "{} vs {bx}", b(&mut s, "y:x"));
    assert!((b(&mut s, "y:_cons") - bc).abs() < 0.03);
    // posterior mean of sigma2 = SSR/(n-k-2) under the Jeffreys prior
    let ssr = rmse * rmse * 78.0;
    assert!((b(&mut s, "sigma2") / (ssr / 76.0) - 1.0).abs() < 0.05);
    // default priors with MH: close to OLS too
    s.run("bayes, mcmcsize(2000) burnin(500) rseed(2): regress y x");
    assert!((b(&mut s, "y:x") - bx).abs() < 0.4 * sex);
    assert_eq!(s.scalar("e(mcmcsize)"), 2000.0);
    let out = s.run("display e(prefix) \" \" e(cmdname)");
    assert!(out.contains("bayes regress"), "{out}");
}

#[test]
fn bayes_logit_poisson_close_to_ml() {
    let mut s = Session::new();
    data(&mut s, 200);
    s.run("logit b x");
    let (bx, se) = (b(&mut s, "x"), s.scalar("_se[x]"));
    s.run("bayes, prior({b:}, flat) mcmcsize(2000) burnin(500) rseed(4): logit b x");
    assert!((b(&mut s, "b:x") - bx).abs() < 0.3 * se, "{} vs {bx}", b(&mut s, "b:x"));
    assert!((sd(&mut s, "b:x") / se - 1.0).abs() < 0.25);
    s.run("poisson k x");
    let (bx, se) = (b(&mut s, "x"), s.scalar("_se[x]"));
    s.run("bayes, mcmcsize(2000) burnin(500) rseed(4): poisson k x");
    assert!((b(&mut s, "k:x") - bx).abs() < 0.3 * se);
}

#[test]
fn postestimation_summaries_and_tests() {
    let mut s = Session::new();
    data(&mut s, 60);
    s.run("bayes, mcmcsize(1000) burnin(300) rseed(9) nchains(2): regress y x");
    s.run("estimates store full");
    let out = s.run("bayesstats summary {y:x} (ratio: {y:x}/{y:_cons})");
    assert!(out.contains("ratio") && out.contains("Posterior summary"), "{out}");
    let out = s.run("bayesstats ess");
    assert!(out.contains("ESS") && out.contains("sigma2"), "{out}");
    let out = s.run("bayesstats grubin");
    assert!(out.contains("Rc"), "{out}");
    assert!(s.scalar("r(Rc_max)") < 1.2);
    let p = {
        s.run("bayestest interval {y:x}, lower(0)");
        s.scalar("el(r(summary), 1, 1)")
    };
    assert!(p > 0.95 && p <= 1.0, "{p}");
    s.run("bayes, mcmcsize(1000) burnin(300) rseed(9): regress y");
    s.run("estimates store null");
    let out = s.run("bayesstats ic full null");
    assert!(out.contains("DIC") && out.contains("log(BF)"), "{out}");
    s.run("bayestest model full null");
    assert!(s.scalar("el(r(test), 1, 3)") > 0.9);
    s.run("estimates restore full");
    for g in ["diagnostics {y:x}", "trace {sigma2}", "ac {y:x}", "histogram {y:x}", "kdensity {y:x}", "cusum {y:x}", "matrix {y:x} {sigma2}"] {
        s.run(&format!("bayesgraph {g}"));
    }
}

#[test]
fn dirichlet_prior_and_multilevel_gibbs() {
    let mut s = Session::new();
    // prior-only sampling: Dirichlet(1, 2, 3) means 1/6, 1/3, 1/2
    s.run("bayesmh, prior({p1 p2 p3}, dirichlet(1,2,3)) mcmcsize(3000) burnin(300) rseed(1)");
    assert!((b(&mut s, "p1") - 1.0 / 6.0).abs() < 0.03);
    assert!((b(&mut s, "p3") - 0.5).abs() < 0.03);
    // random intercepts: Gibbs updates for the effects and their variance
    data(&mut s, 80);
    s.run("mixed y x || id:");
    let bx = b(&mut s, "y:x");
    let out = s.run("bayes, mcmcsize(1500) burnin(300) rseed(3): mixed y x || id:");
    assert!(out.contains("Gibbs") && out.contains("U0"), "{out}");
    assert!((b(&mut s, "y:x") - bx).abs() < 0.08, "{} vs {bx}", b(&mut s, "y:x"));
    // the effects are kept in the simulation results
    let out = s.run("bayesstats summary {U0}");
    assert!(out.contains("U0[1]"), "{out}");
}

#[test]
fn predictions_selection_and_evaluators() {
    let mut s = Session::new();
    data(&mut s, 60);
    s.run("bayes, mcmcsize(400) burnin(200) rseed(1): regress y x");
    s.run("bayespredict pm, rseed(2)");
    s.run("quietly summarize pm");
    let pm = s.scalar("r(mean)");
    s.run("quietly summarize y");
    assert!((pm - s.scalar("r(mean)")).abs() < 0.15);
    s.run("bayespredict lo hi, cri");
    s.run("bayesreps yr, nreps(2)");
    assert!(s.has_var("yr2") && s.has_var("hi"));
    let out = s.run("bayesstats ppvalues (mn: @mean({_ysim}))");
    assert!(out.contains("P(T>=T_obs)"), "{out}");
    let p = s.scalar("el(r(ppvalues), 1, 4)");
    assert!(p > 0.05 && p < 0.95, "{p}");
    // variable selection: the irrelevant predictor is shrunk
    s.run("gen double w = mod(_n*7, 13)/6 - 1");
    s.run("bayesselect y x w, mcmcsize(1500) burnin(300) rseed(1) prior(spikeslab)");
    let pip_x = s.scalar("el(e(inclusion), 1, 1)");
    let pip_w = s.scalar("el(e(inclusion), 1, 2)");
    assert!(pip_x > 0.7 && pip_w < 0.5, "{pip_x} {pip_w}");
    s.run("bayesselect y x w, mcmcsize(800) burnin(200) rseed(1)");
    assert!(b(&mut s, "y:x") > 0.3);
    // user log-likelihood program: normal with unit variance, as dnormal()
    s.script("program lnormal1\n args lnf xb\n quietly replace `lnf' = lnnormalden($MH_y1 - `xb')\nend");
    s.run("bayesmh y, llevaluator(lnormal1) prior({y:_cons}, normal(0,100)) mcmcsize(600) burnin(200) rseed(4)");
    let m1 = b(&mut s, "y:_cons");
    s.run("bayesmh y, likelihood(normal(1)) prior({y:_cons}, normal(0,100)) mcmcsize(600) burnin(200) rseed(4)");
    assert!((m1 - b(&mut s, "y:_cons")).abs() < 0.12);
}

#[test]
fn var_postestimation() {
    let mut s = Session::new();
    s.script(
        "set obs 80
         gen t = _n
         tsset t
         gen double e1 = invnormal((mod(_n*53, 97) + 0.5)/97)
         gen double e2 = invnormal((mod(_n*29, 89) + 0.5)/89)
         gen double y1 = e1
         gen double y2 = e2
         replace y1 = 0.5*y1[_n-1] + e1 in 2/l
         replace y2 = 0.3*y1[_n-1] + 0.2*y2[_n-1] + e2 in 2/l",
    );
    s.run("bayes, mcmcsize(400) burnin(200) rseed(1): var y1 y2, lags(1)");
    s.run("bayesvarstable");
    assert!(s.scalar("r(stable_prob)") > 0.9);
    s.run("bayesirf create a, step(4)");
    let out = s.run("bayesirf table irf, impulse(y1) response(y1)");
    // irf at step 0 of a variable on itself is 1
    assert!(out.contains("Step"), "{out}");
    s.run("bayesirf graph oirf, impulse(y1) response(y2)");
    s.run("bayesfcast compute f_, step(3)");
    assert_eq!(s.n(), 83);
    s.run("bayesfcast graph f_y1");
}

/// Native likelihoods of the prefix agree with maximum likelihood under flat priors.
#[test]
fn prefix_native_likelihoods_match_ml() {
    let mut s = Session::new();
    data(&mut s, 300);
    s.script(
        "gen double lo = floor(y)
         gen double hi = lo + 1
         gen double tt = exp(0.5 - 0.4*x) * (-ln(u))^(1/1.5)
         gen byte dd = mod(_n, 5) > 0
         gen double z = mod(_n*11, 23)/11 - 1
         gen double kz = cond(z > 0.4, 0, k)",
    );
    let check = |s: &mut Session, ml: &str, bayes: &str, ml_name: &str, b_name: &str| {
        s.run(ml);
        let (bx, se) = (b(s, ml_name), s.scalar(&format!("_se[{ml_name}]")));
        s.run(&format!("bayes, prior({{{}}}, flat) mcmcsize(1500) burnin(500) rseed(2): {bayes}", b_name.split(':').next().unwrap().to_string() + ":"));
        let bb = b(s, b_name);
        assert!((bb - bx).abs() < 0.35 * se, "{bayes}: {bb} vs {bx} (se {se})");
    };
    check(&mut s, "intreg lo hi x", "intreg lo hi x", "x", "lo:x");
    check(&mut s, "zip kz x, inflate(z)", "zip kz x, inflate(z)", "kz:x", "kz:x");
    s.run("stset tt, failure(dd)");
    check(&mut s, "streg x, dist(weibull)", "streg x, dist(weibull)", "_t:x", "_t:x");
    check(&mut s, "qreg y x", "qreg y x", "x", "y:x");
}

#[test]
fn mvreg_inverse_wishart_gibbs() {
    let mut s = Session::new();
    data(&mut s, 200);
    s.run("gen double w = 2 - 0.5*x + invnormal(mod(_n*29, 89)/89 + 0.005)");
    s.run("mvreg y w = x");
    let s11 = s.scalar("el(e(Sigma), 1, 1)");
    let s22 = s.scalar("el(e(Sigma), 2, 2)");
    let out = s.run("bayes, mcmcsize(1000) burnin(300) rseed(5): mvreg y w = x");
    assert!(out.contains("Gibbs") && out.contains("Sigma_2_1"), "{out}");
    assert!((b(&mut s, "Sigma_1_1") / s11 - 1.0).abs() < 0.15);
    assert!((b(&mut s, "Sigma_2_2") / s22 - 1.0).abs() < 0.15);
}
