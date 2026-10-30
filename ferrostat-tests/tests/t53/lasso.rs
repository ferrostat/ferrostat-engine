#[allow(unused_imports)]
use ferrostat_tests::*;

/// Sparse linear design: y depends on x1 and x2 only; covariates are
/// standardized (population sd) so penalties act on the stored scale.
fn sparse(s: &mut Session, n: usize, p: usize) {
    s.script(&format!(
        "clear
         set obs {n}
         set seed 53
         forvalues j = 1/{p} {{
             gen double x`j' = rnormal()
             quietly summarize x`j'
             replace x`j' = (x`j' - r(mean)) / (r(sd) * sqrt((r(N)-1)/r(N)))
         }}
         gen double d = 0.5*x1 + rnormal()
         gen double y = 1 + 2*x1 - 1.5*x2 + 0.7*d + rnormal()
         gen byte b = (y + rnormal()) > 1
         gen double c = rpoisson(exp(0.3 + 0.4*x1 - 0.3*x2))"
    ));
}

fn coef(s: &mut Session, name: &str) -> f64 {
    s.scalar(&format!("_b[{name}]"))
}

#[test]
fn tiny_lambda_reproduces_least_squares() {
    let mut s = Session::new();
    sparse(&mut s, 120, 6);
    s.run("regress y x1-x6");
    let ols: Vec<f64> = (1..=6).map(|j| coef(&mut s, &format!("x{j}"))).collect();
    let c0 = coef(&mut s, "_cons");
    s.run("lasso linear y x1-x6, selection(none) grid(40, ratio(1e-8)) stop(0)");
    s.run("lassoselect id = 40");
    for j in 1..=6 {
        assert_close(coef(&mut s, &format!("x{j}")), ols[j - 1], 1e-5);
    }
    assert_close(coef(&mut s, "_cons"), c0, 1e-5);
    // lambda above lambda_max: nothing selected
    s.run("lassoselect id = 1");
    assert_eq!(s.scalar("e(k_nonzero_sel)"), 0.0);
}

#[test]
fn kkt_conditions_hold_at_the_selected_lambda() {
    let mut s = Session::new();
    sparse(&mut s, 150, 12);
    s.run("lasso linear y x1-x12, rseed(12345) folds(5)");
    let lam = s.scalar("e(lambda_sel)");
    assert!(lam > 0.0 && lam < s.scalar("e(lambda_max)"));
    let sel = s.scalar("e(k_nonzero_sel)");
    assert!(sel >= 2.0);
    s.run("predict double xbp, xb penalized");
    s.run("gen double r = y - xbp");
    // residuals have mean zero (unpenalized intercept)
    s.run("quietly summarize r");
    assert!(s.scalar("r(mean)").abs() < 1e-6);
    let bnames = s.run("matrix list e(b)");
    for j in 1..=12 {
        s.run(&format!("gen double g{j} = x{j} * r"));
        s.run(&format!("quietly summarize g{j}"));
        let g = s.scalar("r(mean)");
        let name = format!("x{j}");
        let nonzero = bnames.split_whitespace().any(|w| w == name);
        if nonzero {
            let b = coef(&mut s, &name);
            assert_close(g, lam * b.signum(), 2e-4);
        } else {
            assert!(g.abs() <= lam + 2e-4, "x{j}: |{g}| > {lam}");
        }
    }
    // the same seed gives the same selection
    s.run("lasso linear y x1-x12, rseed(12345) folds(5)");
    assert_close(s.scalar("e(lambda_sel)"), lam, 1e-12);
}

#[test]
fn postestimation_tools() {
    let mut s = Session::new();
    sparse(&mut s, 100, 8);
    let out = s.run("lasso linear y x1-x8, rseed(7)");
    assert!(out.contains("Cross-validation") && out.contains("selected lambda"), "{out}");
    let k = s.run("lassoknots");
    assert!(k.contains("A x1"), "{k}");
    let c = s.run("lassocoef");
    assert!(c.contains("x1") && c.contains("x2"), "{c}");
    s.run("lassocoef, display(coef, postselection)");
    // postselection predictions equal OLS on the selected covariates
    let vars = s.run("display e(post_sel_vars)").trim().to_string();
    s.run("predict double yp, postselection");
    s.run(&format!("regress y {vars}"));
    s.run("predict double yo");
    s.run("gen double dd = abs(yp - yo)");
    s.run("quietly summarize dd");
    assert!(s.scalar("r(max)") < 1e-6);
    // lassogof penalized MSE equals the mean squared residual
    s.run("lasso linear y x1-x8, rseed(7)");
    s.run("estimates store cvfit");
    s.run("predict double yh");
    s.run("gen double sq = (y - yh)^2");
    s.run("quietly summarize sq");
    let mse = s.scalar("r(mean)");
    let g = s.run("lassogof");
    assert!(g.contains("MSE"), "{g}");
    assert_close(s.scalar("el(r(table),1,1)"), mse, 1e-8);
    s.run("lassogof, postselection");
    s.run("lassoinfo");
    s.run("lassoinfo cvfit");
    s.run("cvplot");
    s.run("coefpath");
    s.run("lasso linear y x1-x8, selection(bic)");
    assert!(s.scalar("e(k_nonzero_sel)") >= 2.0);
    s.run("bicplot");
    s.run("lassoselect lambda = 0.1");
    assert!(s.run("lassoknots, display(nonzero bic l1)").contains("BIC"));
}

#[test]
fn logit_and_poisson_lassos_reduce_to_ml() {
    let mut s = Session::new();
    sparse(&mut s, 200, 4);
    s.run("logit b x1-x4");
    let lb: Vec<f64> = (1..=4).map(|j| coef(&mut s, &format!("x{j}"))).collect();
    s.run("lasso logit b x1-x4, selection(none) grid(30, ratio(1e-8)) stop(0)");
    s.run("lassoselect id = 30");
    for j in 1..=4 {
        assert_close(coef(&mut s, &format!("x{j}")), lb[j - 1], 1e-4);
    }
    s.run("predict double p");
    s.run("quietly summarize p");
    assert!(s.scalar("r(min)") > 0.0 && s.scalar("r(max)") < 1.0);
    s.run("poisson c x1-x4");
    let pb: Vec<f64> = (1..=4).map(|j| coef(&mut s, &format!("x{j}"))).collect();
    s.run("lasso poisson c x1-x4, selection(none) grid(30, ratio(1e-8)) stop(0)");
    s.run("lassoselect id = 30");
    for j in 1..=4 {
        assert_close(coef(&mut s, &format!("x{j}")), pb[j - 1], 1e-4);
    }
    s.run("probit b x1-x4");
    let qb = coef(&mut s, "x1");
    s.run("lasso probit b x1-x4, selection(none) grid(30, ratio(1e-8)) stop(0)");
    s.run("lassoselect id = 30");
    assert_close(coef(&mut s, "x1"), qb, 1e-4);
    // cross-validated logit lasso keeps the strong predictors
    s.run("lasso logit b x1-x4, rseed(3)");
    let v = s.run("display e(post_sel_vars)");
    assert!(v.contains("x1"), "{v}");
}

#[test]
fn elastic_net_and_square_root_lasso() {
    let mut s = Session::new();
    sparse(&mut s, 120, 6);
    s.run("regress y x1-x6");
    let ols = coef(&mut s, "x3");
    // pure ridge with a tiny penalty is least squares
    s.run("elasticnet linear y x1-x6, alpha(0) selection(none) grid(30, ratio(1e-9)) stop(0)");
    s.run("lassoselect alpha = 0 lambda = 0");
    assert_close(coef(&mut s, "x3"), ols, 1e-4);
    // alpha(1) is the lasso
    s.run("lasso linear y x1-x6, selection(none) grid(20) stop(0)");
    s.run("lassoselect id = 10");
    let l10 = coef(&mut s, "x1");
    s.run("elasticnet linear y x1-x6, alpha(1) selection(none) grid(20) stop(0)");
    s.run("lassoselect id = 10");
    assert_close(coef(&mut s, "x1"), l10, 1e-8);
    let out = s.run("elasticnet linear y x1-x6, alpha(0.5 1) rseed(2)");
    assert!(out.contains("alpha"), "{out}");
    assert!(s.scalar("e(alpha_sel)") >= 0.5);
    // square-root lasso: pivotal plug-in keeps x1 and x2
    sparse(&mut s, 300, 20);
    s.run("sqrtlasso y x1-x20, selection(plugin)");
    let v = s.run("display e(post_sel_vars)");
    assert!(v.contains("x1") && v.contains("x2"), "{v}");
    assert!(s.scalar("e(k_nonzero_sel)") <= 6.0);
    s.run("sqrtlasso y x1-x20, rseed(4) folds(4)");
    assert!(s.scalar("e(lambda_sel)") > 0.0);
}

#[test]
fn plugin_and_adaptive_selection() {
    let mut s = Session::new();
    sparse(&mut s, 400, 30);
    s.run("lasso linear y x1-x30, selection(plugin)");
    let v = s.run("display e(post_sel_vars)");
    assert!(v.contains("x1") && v.contains("x2"), "{v}");
    assert!(s.scalar("e(k_nonzero_sel)") <= 6.0);
    s.run("lasso linear y (x3) x1 x2 x4-x30, selection(plugin, homoskedastic)");
    let v = s.run("display e(post_sel_vars)");
    assert!(v.contains("x3"), "always-included covariate kept: {v}");
    s.run("lasso linear y x1-x30, selection(adaptive) rseed(5)");
    let k = s.scalar("e(k_nonzero_sel)");
    assert!((2.0..=12.0).contains(&k));
    s.run("lasso logit b x1-x30, selection(plugin)");
    s.run("lasso poisson c x1-x30, selection(plugin)");
}

#[test]
fn double_selection_and_partialing_out_with_fixed_controls() {
    let mut s = Session::new();
    sparse(&mut s, 200, 5);
    // with every control always included, double selection is OLS
    s.run("regress y d x1-x5, vce(robust)");
    let b = coef(&mut s, "d");
    let se = s.scalar("_se[d]");
    let out = s.run("dsregress y d, controls((x1-x5))");
    assert!(out.contains("Double-selection linear model"), "{out}");
    assert_close(coef(&mut s, "d"), b, 1e-8);
    assert_close(s.scalar("_se[d]"), se, 1e-8);
    // partialing-out (Frisch-Waugh-Lovell) gives the same point estimate
    s.run("poregress y d, controls((x1-x5))");
    assert_close(coef(&mut s, "d"), b, 1e-8);
    // logit: double selection with fixed controls is logit
    s.run("logit b d x1-x5, vce(robust)");
    let lb = coef(&mut s, "d");
    s.run("dslogit b d, controls((x1-x5))");
    assert_close(coef(&mut s, "d"), lb, 1e-6);
    s.run("poisson c d x1-x5");
    let pb = coef(&mut s, "d");
    s.run("dspoisson c d, controls((x1-x5))");
    assert_close(coef(&mut s, "d"), pb, 1e-6);
    // partialing-out logit/Poisson solve the same first-order condition
    s.run("pologit b d, controls((x1-x5))");
    assert_close(coef(&mut s, "d"), lb, 1e-5);
    s.run("popoisson c d, controls((x1-x5))");
    assert_close(coef(&mut s, "d"), pb, 1e-5);
}

#[test]
fn selection_based_inference_recovers_the_effect() {
    let mut s = Session::new();
    sparse(&mut s, 400, 40);
    s.run("dsregress y d, controls(x1-x40)");
    let b = coef(&mut s, "d");
    let se = s.scalar("_se[d]");
    assert!((b - 0.7).abs() < 4.0 * se, "{b} {se}");
    let info = s.run("lassoinfo");
    assert!(info.contains("plugin"), "{info}");
    s.run("lassocoef (., for(y)) (., for(d))");
    s.run("poregress y d, controls(x1-x40) selection(cv) rseed(1)");
    assert!((coef(&mut s, "d") - 0.7).abs() < 4.0 * s.scalar("_se[d]"));
    s.run("lassoknots, for(d)");
    s.run("lassoselect id = 3, for(d)");
    s.run("poregress, reestimate");
    s.run("xporegress y d, controls(x1-x40) xfolds(4) rseed(9)");
    assert!((coef(&mut s, "d") - 0.7).abs() < 4.0 * s.scalar("_se[d]"));
    assert_eq!(s.scalar("e(n_xfolds)"), 4.0);
    s.run("xporegress y d, controls(x1-x40) xfolds(3) resample(2) rseed(9)");
    s.run("xpologit b d, controls(x1-x40) xfolds(3) rseed(9)");
    s.run("xpopoisson c d, controls(x1-x40) xfolds(3) rseed(9)");
    s.fails("predict yy");
}

#[test]
fn partialing_out_iv_matches_two_stage_least_squares() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 300
         set seed 9
         gen double x1 = rnormal()
         gen double x2 = rnormal()
         gen double z = rnormal()
         gen double u = rnormal()
         gen double dn = 0.8*z + 0.5*u + 0.3*x1 + rnormal()
         gen double y = 1 + 0.5*dn + x1 - x2 + u",
    );
    s.run("ivregress 2sls y x1 x2 (dn = z)");
    let b = coef(&mut s, "dn");
    s.run("poivregress y (dn = z), controls((x1 x2))");
    assert_close(coef(&mut s, "dn"), b, 1e-7);
    s.run("xpoivregress y (dn = z), controls(x1 x2) xfolds(3) rseed(2)");
    assert!((coef(&mut s, "dn") - 0.5).abs() < 0.3);
}
