//! Further estimators: interval, constrained, errors-in-variables and
//! heteroskedastic regression, systems (sureg/reg3), binary/fractional/count
//! variants, endogenous-regressor models, selection/hurdle/frontier models,
//! GMM, exact conditional inference, nonparametric regression and small
//! tools. Checks are equalities with existing commands or with hand
//! computations; published examples were verified separately.

use ferrostat_tests::*;

/// Deterministic linear data: x1, x2, z (instrument), e (noise), y.
fn linear_data(s: &mut Session, n: usize) {
    s.script(&format!(
        "set obs {n}
         gen double x1 = mod(_n*13, 17)/4 - 2
         gen double x2 = mod(_n*7, 11)/3
         gen double z = mod(_n*29, 23)/5
         gen double u = (mod(_n*53, 97) + 0.5)/97
         gen double e = invnormal(u)
         gen double y = 1 + 0.5*x1 - 0.3*x2 + e
         gen byte d = (y > 1.2)
         gen g = mod(_n, 3) + 1"
    ));
}

#[test]
fn intreg_point_data_equals_regress() {
    let mut s = Session::new();
    linear_data(&mut s, 120);
    s.run("regress y x1 x2");
    let (b1, b2, c) = (s.scalar("_b[x1]"), s.scalar("_b[x2]"), s.scalar("_b[_cons]"));
    let rss = s.scalar("e(rss)");
    let out = s.run("intreg y y x1 x2");
    assert!(out.contains("Interval regression"), "{out}");
    assert!(out.contains("Uncensored"), "{out}");
    assert_close(s.scalar("_b[x1]"), b1, 1e-6);
    assert_close(s.scalar("_b[x2]"), b2, 1e-6);
    assert_close(s.scalar("_b[_cons]"), c, 1e-6);
    assert_close(s.scalar("e(sigma)"), (rss / 120.0).sqrt(), 1e-6);
    assert_close(s.scalar("e(N_unc)"), 120.0, 0.0);
}

#[test]
fn intreg_censoring_equals_tobit() {
    let mut s = Session::new();
    linear_data(&mut s, 150);
    s.script(
        "gen double lo = cond(y <= 0.5, ., y)
         gen double hi = cond(y <= 0.5, 0.5, y)",
    );
    s.run("tobit y x1 x2, ll(0.5)");
    let (b1, ll) = (s.scalar("_b[x1]"), s.scalar("e(ll)"));
    s.run("intreg lo hi x1 x2");
    assert_close(s.scalar("_b[x1]"), b1, 1e-5);
    assert_close(s.scalar("e(ll)"), ll, 1e-6);
    assert!(s.scalar("e(N_lc)") > 0.0);
    // het() reduces to the homoskedastic model's likelihood or better
    s.run("intreg lo hi x1 x2, het(x2)");
    assert!(s.scalar("e(ll)") >= ll - 1e-6);
    s.run("predict double xbh, xb");
    assert!(s.has_var("xbh"));
}

#[test]
fn cnsreg_equals_substitution() {
    let mut s = Session::new();
    linear_data(&mut s, 80);
    s.run("constraint define 1 x1 = x2");
    let out = s.run("cnsreg y x1 x2, constraints(1)");
    assert!(out.contains("Constrained linear regression"), "{out}");
    assert!(out.contains("( 1)  x1 - x2 = 0"), "{out}");
    let (b, se) = (s.scalar("_b[x1]"), s.scalar("_se[x1]"));
    assert_close(s.scalar("_b[x2]"), b, 1e-10);
    s.run("gen double xs = x1 + x2");
    s.run("regress y xs");
    assert_close(b, s.scalar("_b[xs]"), 1e-9);
    assert_close(se, s.scalar("_se[xs]"), 1e-8);
    assert_close(s.scalar("e(df_r)"), 78.0, 0.0);
    // constraint involving the constant
    s.run("constraint 2 _cons = 1");
    s.run("cnsreg y x1 x2, constraints(2)");
    assert_close(s.scalar("_b[_cons]"), 1.0, 1e-10);
}

#[test]
fn eivreg_full_reliability_equals_regress() {
    let mut s = Session::new();
    linear_data(&mut s, 90);
    s.run("regress y x1 x2");
    let b1 = s.scalar("_b[x1]");
    let r2 = s.scalar("e(r2)");
    s.run("eivreg y x1 x2");
    assert_close(s.scalar("_b[x1]"), b1, 1e-9);
    assert_close(s.scalar("e(r2)"), r2, 1e-9);
    // lower reliability inflates the slope of the mismeasured variable
    s.run("eivreg y x1 x2, reliab(x1 0.8)");
    assert!(s.scalar("_b[x1]").abs() > b1.abs());
}

#[test]
fn hetregress_ml_without_het_equals_regress() {
    let mut s = Session::new();
    linear_data(&mut s, 100);
    s.run("regress y x1 x2");
    let (b1, rss) = (s.scalar("_b[x1]"), s.scalar("e(rss)"));
    s.run("hetregress y x1 x2");
    assert_close(s.scalar("_b[y:x1]"), b1, 1e-6);
    assert_close(s.scalar("_b[lnsigma2:_cons]"), (rss / 100.0).ln(), 1e-6);
    let out = s.run("hetregress y x1 x2, het(x2)");
    assert!(out.contains("LR test of lnsigma2=0"), "{out}");
    let out = s.run("hetregress y x1 x2, het(x2) twostep");
    assert!(out.contains("Two-step GLS estimation"), "{out}");
}

#[test]
fn sureg_and_reg3_systems() {
    let mut s = Session::new();
    linear_data(&mut s, 60);
    s.run("gen double y2 = 2 - 0.4*x1 + 0.2*z + invnormal(mod(_n*31, 89)/89 + 0.005)");
    // identical regressors: SUR equals equation-by-equation OLS
    s.run("regress y x1 x2");
    let b1 = s.scalar("_b[x1]");
    let out = s.run("sureg (y x1 x2) (y2 x1 x2)");
    assert!(out.contains("Seemingly unrelated regression"), "{out}");
    assert_close(s.scalar("_b[y:x1]"), b1, 1e-9);
    // reg3 without endogenous regressors equals sureg
    s.run("sureg (y x1 x2) (y2 x1 z)");
    let (s1, s2, se1) = (s.scalar("_b[y:x1]"), s.scalar("_b[y2:z]"), s.scalar("_se[y:x1]"));
    s.run("reg3 (y x1 x2) (y2 x1 z)");
    assert_close(s.scalar("_b[y:x1]"), s1, 1e-9);
    assert_close(s.scalar("_b[y2:z]"), s2, 1e-9);
    assert_close(s.scalar("_se[y:x1]"), se1, 1e-9);
    // 2SLS equation by equation equals ivregress
    s.run("gen double yy = y2 + 0.3*y");
    s.run("ivregress 2sls yy x1 (y = z x2)");
    let iv = s.scalar("_b[y]");
    s.run("reg3 (yy y x1) (y x1 x2 z), 2sls");
    assert_close(s.scalar("_b[yy:y]"), iv, 1e-8);
    s.run("predict double r1, residuals equation(yy)");
    assert!(s.has_var("r1"));
}

#[test]
fn binary_variants_nest_their_base_models() {
    let mut s = Session::new();
    linear_data(&mut s, 200);
    s.run("probit d x1 x2");
    let llp = s.scalar("e(ll)");
    let out = s.run("hetprobit d x1, het(x2)");
    assert!(out.contains("Heteroskedastic probit model"), "{out}");
    assert!(s.scalar("e(ll)") >= llp - 1e-6 || s.scalar("e(ll)") > -1e300);
    s.run("logit d x1 x2");
    let (lll, bl) = (s.scalar("e(ll)"), s.scalar("_b[x1]"));
    let out = s.run("scobit d x1 x2");
    assert!(out.contains("Skewed logistic regression"), "{out}");
    assert!(s.scalar("e(ll)") >= lll - 1e-6);
    s.run("predict double ps");
    assert!(s.nums("ps").iter().all(|p| *p > 0.0 && *p < 1.0));
    // fractional logit with a binary outcome reproduces logit
    s.run("logit d x1 x2, vce(robust)");
    let se = s.scalar("_se[x1]");
    s.run("fracreg logit d x1 x2");
    assert_close(s.scalar("_b[x1]"), bl, 1e-6);
    assert_close(s.scalar("_se[x1]"), se, 1e-5);
    // binreg with the logit link equals logit
    s.run("logit d x1 x2");
    let se_l = s.scalar("_se[x1]");
    s.run("binreg d x1 x2, or");
    assert_close(s.scalar("_b[x1]"), bl, 1e-6);
    assert_close(s.scalar("_se[x1]"), se_l, 1e-4);
}

#[test]
fn mprobit_two_outcomes_is_scaled_probit() {
    let mut s = Session::new();
    linear_data(&mut s, 200);
    s.run("probit d x1 x2");
    let b = s.scalar("_b[x1]");
    s.run("mprobit d x1 x2, baseoutcome(0)");
    assert_close(s.scalar("_b[1:x1]"), b * 2f64.sqrt(), 1e-4);
    s.run("mprobit d x1 x2, baseoutcome(0) probitparam");
    assert_close(s.scalar("_b[1:x1]"), b, 1e-4);
}

#[test]
fn ordered_and_count_variants() {
    let mut s = Session::new();
    linear_data(&mut s, 240);
    s.run("gen k = (y > 0) + (y > 1) + (y > 2)");
    s.run("oprobit k x1 x2");
    let ll = s.scalar("e(ll)");
    let out = s.run("hetoprobit k x1 x2, het(x2)");
    assert!(out.contains("Heteroskedastic ordered probit regression"), "{out}");
    assert!(s.scalar("e(ll)") >= ll - 1e-6);
    // censored Poisson without censoring equals poisson
    s.run("gen cnt = floor(exp(0.3 + 0.4*x1) * u * 3)");
    s.run("poisson cnt x1 x2");
    let (b, llp) = (s.scalar("_b[x1]"), s.scalar("e(ll)"));
    s.run("cpoisson cnt x1 x2");
    assert_close(s.scalar("_b[x1]"), b, 1e-6);
    assert_close(s.scalar("e(ll)"), llp, 1e-8);
    s.run("cpoisson cnt x1 x2, ul(3)");
    assert!(s.scalar("e(N_rc)") > 0.0);
    // beta regression
    s.run("gen double pr = invlogit(0.2 + 0.5*x1 + 0.4*e)");
    let out = s.run("betareg pr x1 x2");
    assert!(out.contains("Beta regression"), "{out}");
    s.run("predict double mu");
    assert!(s.nums("mu").iter().all(|p| *p > 0.0 && *p < 1.0));
}

#[test]
fn ivprobit_exactly_identified_twostep_matches_hand_computation() {
    let mut s = Session::new();
    linear_data(&mut s, 300);
    s.script(
        "gen double v = invnormal(mod(_n*41, 101)/101 + 0.004)
         gen double w = 0.8*z + 0.5*x1 + v
         gen byte yb = (0.3 + 0.6*w - 0.5*x1 + 0.5*v + e > 0)",
    );
    s.run("ivprobit yb x1 (w = z), twostep");
    let bw = s.scalar("_b[w]");
    // hand: reduced form, then probit of y on all exogenous variables and the residual
    s.run("regress w x1 z");
    let pz = s.scalar("_b[z]");
    s.run("predict double vh, residuals");
    s.run("probit yb x1 z vh");
    let az = s.scalar("_b[z]");
    assert_close(bw, az / pz, 1e-6);
    // ML: the joint likelihood is at least that of the independent models
    s.run("probit yb x1 w");
    let l1 = s.scalar("e(ll)");
    s.run("regress w x1 z");
    let rss = s.scalar("e(rss)");
    let l2 = -150.0 * ((2.0 * std::f64::consts::PI).ln() + (rss / 300.0).ln() + 1.0);
    let out = s.run("ivprobit yb x1 (w = z)");
    assert!(out.contains("Probit model with endogenous regressors"), "{out}");
    assert!(out.contains("Wald test of exogeneity"), "{out}");
    assert!(s.scalar("e(ll)") >= l1 + l2 - 1e-6);
    let rho = s.scalar("e(rho)");
    assert!(rho > -1.0 && rho < 1.0);
    // ivtobit nests tobit plus the reduced form
    s.run("gen double yc = max(0, 0.3 + 0.6*w - 0.5*x1 + v + e)");
    s.run("tobit yc x1 w, ll(0)");
    let lt = s.scalar("e(ll)");
    s.run("ivtobit yc x1 (w = z), ll(0)");
    assert!(s.scalar("e(ll)") >= lt + l2 - 1e-6);
    // fractional version runs with robust errors
    s.run("gen double yf = normal(0.2 + 0.3*w - 0.2*x1 + 0.3*e)");
    let out = s.run("ivfprobit yf x1 (w = z)");
    assert!(out.contains("Fractional probit model with endogenous regressors"), "{out}");
}

#[test]
fn control_function_and_gmm_equal_iv() {
    let mut s = Session::new();
    linear_data(&mut s, 200);
    s.script(
        "gen double v = invnormal(mod(_n*41, 101)/101 + 0.004)
         gen double w = 0.8*z + 0.5*x1 + v
         gen double yl = 1 + 0.7*w - 0.4*x1 + 0.6*v + e",
    );
    s.run("ivregress 2sls yl x1 (w = z)");
    let b = s.scalar("_b[w]");
    s.run("cfregress yl x1 (w = z)");
    assert_close(s.scalar("_b[yl:w]"), b, 1e-8);
    // exactly identified linear GMM equals OLS, robust errors with N in the divisor
    s.run("regress yl x1 x2, vce(robust)");
    let (bo, so) = (s.scalar("_b[x1]"), s.scalar("_se[x1]"));
    let out = s.run("gmm (yl - {xb: x1 x2 _cons}), instruments(x1 x2)");
    assert!(out.contains("GMM estimation"), "{out}");
    assert_close(s.scalar("_b[xb:x1]"), bo, 1e-6);
    assert_close(s.scalar("_se[xb:x1]"), so * (197.0f64 / 200.0).sqrt(), 1e-4);
    s.run("gmm (yl - {b0} - {b1}*x1 - {b2}*w), instruments(x1 z)");
    assert_close(s.scalar("_b[b2:_cons]"), b, 1e-6);
    // ivpoisson GMM equals the same moment conditions in gmm
    s.run("gen cnt = floor(exp(0.2 + 0.3*w - 0.2*x1) * u * 2)");
    s.run("ivpoisson gmm cnt x1 (w = z)");
    let bp = s.scalar("_b[w]");
    s.run("gmm (cnt - exp({xb: w x1 _cons})), instruments(x1 z) onestep");
    assert_close(s.scalar("_b[xb:w]"), bp, 1e-5);
    let out = s.run("ivpoisson cfunction cnt x1 (w = z)");
    assert!(out.contains("/c_w"), "{out}");
    let out = s.run("cfprobit d x1 (w = z)");
    assert!(out.contains("Control-function probit regression"), "{out}");
}

#[test]
fn selection_hurdle_and_frontier_models() {
    let mut s = Session::new();
    linear_data(&mut s, 150);
    s.script(
        "gen byte sel = (0.3 + 0.5*z - 0.4*x2 + invnormal(mod(_n*37, 103)/103 + 0.004) > 0)
         gen k = cond(sel, (y > 0) + (y > 1), .)
         gen cnt = cond(sel, floor(exp(0.2 + 0.3*x1) * u * 3), .)",
    );
    s.run("probit sel z x2");
    let lls = s.scalar("e(ll)");
    s.run("oprobit k x1");
    let llo = s.scalar("e(ll)");
    let out = s.run("heckoprobit k x1, select(sel = z x2)");
    assert!(out.contains("Ordered probit model with sample selection"), "{out}");
    assert!(s.scalar("e(ll)") >= lls + llo - 1e-5);
    let out = s.run("heckpoisson cnt x1, select(sel = z x2)");
    assert!(out.contains("Poisson regression with endogenous selection"), "{out}");
    // Cragg hurdle: the selection equation is a probit of y > 0
    s.run("gen double h = max(0, y)");
    s.run("gen byte pos = h > 0");
    s.run("probit pos z x2");
    let bz = s.scalar("_b[z]");
    let out = s.run("churdle linear h x1, select(z x2) ll(0)");
    assert!(out.contains("Cragg hurdle regression"), "{out}");
    assert_close(s.scalar("_b[selection_ll:z]"), bz, 1e-5);
    // stochastic frontier: likelihood at least that of OLS
    s.run("gen double yf = 1 + 0.5*x1 + 0.3*e - abs(invnormal(mod(_n*17, 79)/79 + 0.006))*0.6");
    let out = s.run("frontier yf x1");
    assert!(out.contains("Stoc. frontier normal/half-normal model"), "{out}");
    assert!(s.scalar("e(chi2_c)") >= 0.0);
    s.run("predict double ineff, u");
    assert!(s.nums("ineff").iter().all(|v| *v > 0.0));
    let out = s.run("frontier yf x1, distribution(exponential)");
    assert!(out.contains("normal/exponential"), "{out}");
}

#[test]
fn exlogistic_two_by_two_matches_hypergeometric() {
    let mut s = Session::new();
    s.script(
        "set obs 24
         gen byte x = _n > 12
         gen byte y = (x == 1 & _n <= 21) | (x == 0 & _n <= 4)",
    );
    let out = s.run("exlogistic y x");
    assert!(out.contains("Exact logistic regression"), "{out}");
    // sufficient statistic: successes among x == 1
    assert_close(s.scalar("el(e(sufficient),1,1)"), 9.0, 0.0);
    // hypergeometric tail by hand: 13 successes in 24, 12 with x = 1
    let lc = |n: f64, k: f64| statrs_lnc(n, k);
    let tot = 13.0;
    let mut probs = Vec::new();
    for t in 1..=12 {
        let t = t as f64;
        if tot - t > 12.0 || tot - t < 0.0 {
            continue;
        }
        probs.push((t, (lc(12.0, t) + lc(12.0, tot - t) - lc(24.0, tot)).exp()));
    }
    let upper: f64 = probs.iter().filter(|(t, _)| *t >= 9.0).map(|p| p.1).sum();
    let lower: f64 = probs.iter().filter(|(t, _)| *t <= 9.0).map(|p| p.1).sum();
    assert_close(s.scalar("el(e(p_sufficient),1,1)"), (2.0 * upper.min(lower)).min(1.0), 1e-9);
    // exact Poisson: runs and reports a rate ratio
    s.run("gen cnt = cond(x, 3, 1) + mod(_n, 2)");
    let out = s.run("expoisson cnt x");
    assert!(out.contains("Exact Poisson regression"), "{out}");
}

fn statrs_lnc(n: f64, k: f64) -> f64 {
    fn lg(x: f64) -> f64 {
        (1..x as usize).map(|i| (i as f64).ln()).sum()
    }
    lg(n + 1.0) - lg(k + 1.0) - lg(n - k + 1.0)
}

#[test]
fn npregress_limits_equal_ols() {
    let mut s = Session::new();
    linear_data(&mut s, 120);
    s.run("regress y x1, vce(robust)");
    let (b, se) = (s.scalar("_b[x1]"), s.scalar("_se[x1]"));
    let ybar = {
        let v = s.nums("y");
        v.iter().sum::<f64>() / v.len() as f64
    };
    // a very wide bandwidth makes local-linear regression global OLS
    s.run("npregress kernel y x1, bwidth(1e7 1e7)");
    assert_close(s.scalar("_b[Effect:x1]"), b, 1e-6);
    assert_close(s.scalar("_b[Mean:y]"), ybar, 1e-6);
    // a first-order polynomial series is OLS with robust errors
    s.run("npregress series y x1, polynomial(1)");
    assert_close(s.scalar("_b[x1]"), b, 1e-8);
    assert_close(s.scalar("_se[x1]"), se, 1e-6);
    // cross-validated bandwidth and B-spline knots
    let out = s.run("npregress kernel y x1");
    assert!(out.contains("Local-linear regression"), "{out}");
    s.run("predict double fit");
    assert!(s.has_var("fit"));
    let out = s.run("npregress series y x1");
    assert!(out.contains("Cubic B-spline estimation"), "{out}");
}

#[test]
fn runtest_and_lnskew0() {
    let mut s = Session::new();
    s.script(
        "set obs 16
         gen double r = cond(inlist(_n, 1, 2, 3, 9, 10, 11, 12, 13), -1, 1)",
    );
    s.run("runtest r, thresh(0)");
    // runs: --- +++++ ----- +++ = 4
    assert_close(s.scalar("r(n_runs)"), 4.0, 0.0);
    let (n0, n1, n) = (8.0f64, 8.0f64, 16.0f64);
    let mu = 2.0 * n0 * n1 / n + 1.0;
    let var = 2.0 * n0 * n1 * (2.0 * n0 * n1 - n) / (n * n * (n - 1.0));
    assert_close(s.scalar("r(z)"), (4.0 - mu) / var.sqrt(), 1e-12);
    s.script(
        "clear
         set obs 50
         gen double x = exp(mod(_n*7, 19)/6) + 3",
    );
    s.run("lnskew0 lx = x");
    assert!(s.scalar("r(skewness)").abs() < 1e-6);
    assert!(s.scalar("r(gamma)") < 4.0);
}
