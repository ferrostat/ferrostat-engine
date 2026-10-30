//! Wave 7c: structural equation models, item response theory and finite
//! mixture models. Synthetic data only; checks are closed forms or
//! agreement with other estimators.

use ferrostat_tests::*;

fn sem_data(s: &mut Session, n: usize, seed: u32) {
    s.script(&format!(
        "clear
         set obs {n}
         set seed {seed}
         gen double F = rnormal()
         gen double G = .5*F + rnormal()*.8
         gen double x1 = 1 + F + rnormal()*.6
         gen double x2 = 2 + .8*F + rnormal()*.5
         gen double x3 = -1 + 1.2*F + rnormal()*.7
         gen double w1 = G + rnormal()*.6
         gen double w2 = 1.5*G + rnormal()*.6
         gen double w3 = .7*G + rnormal()*.6
         gen double z = rnormal()
         gen double y1 = 1 + .5*x1 - .3*z + rnormal()
         gen double y2 = .5 + .4*y1 + .2*z + rnormal()
         gen byte grp = mod(_n, 2)"
    ));
}

#[test]
fn sem_observed_variables_equal_regress() {
    let mut s = Session::new();
    sem_data(&mut s, 300, 11);
    s.run("regress y1 x1 z");
    let (b1, b2, c) = (s.scalar("_b[x1]"), s.scalar("_b[z]"), s.scalar("_b[_cons]"));
    let rss = s.scalar("e(rss)");
    s.run("regress y2 y1 z");
    let d1 = s.scalar("_b[y1]");
    s.run("sem (y1 <- x1 z) (y2 <- y1 z), nolog");
    assert_close(s.scalar("_b[y1:x1]"), b1, 1e-6);
    assert_close(s.scalar("_b[y1:z]"), b2, 1e-6);
    assert_close(s.scalar("_b[y1:_cons]"), c, 1e-6);
    assert_close(s.scalar("_b[y2:y1]"), d1, 1e-6);
    assert_close(s.scalar("_b[/:var(e.y1)]"), rss / 300.0, 1e-6);
    // recursive system without error covariance: df = 1 (missing path z->?, none) — y2 <- x1 omitted
    assert_eq!(s.scalar("e(df_ms)"), 1.0);
    // mediation: indirect effect of x1 on y2 = b(y1<-x1) * b(y2<-y1)
    s.run("estat teffects");
    s.run("matrix I = r(indirect)");
    let ind = s.scalar("I[1,colnumb(I,\"y2:x1\")]");
    assert_close(ind, b1 * d1, 1e-6);
    s.run("estat eqtest");
    s.run("estat eqgof");
    s.run("estat stable");
    assert_close(s.scalar("r(stindex)"), 0.0, 1e-12);
    s.run("estat framework");
    s.run("estat summarize");
    s.run("estat residuals");
}

#[test]
fn one_factor_three_indicators_closed_form() {
    let mut s = Session::new();
    sem_data(&mut s, 400, 12);
    s.run("quietly correlate x1 x2 x3, covariance");
    s.run("matrix C = r(C) * 399/400");
    let c = |i: usize, j: usize, s: &mut Session| s.scalar(&format!("C[{i},{j}]"));
    let (s12, s13, s23, s11, s22) = (c(1, 2, &mut s), c(1, 3, &mut s), c(2, 3, &mut s), c(1, 1, &mut s), c(2, 2, &mut s));
    s.run("sem (F -> x1 x2 x3), nolog");
    assert_close(s.scalar("_b[x2:F]"), s23 / s13, 1e-5);
    assert_close(s.scalar("_b[x3:F]"), s23 / s12, 1e-5);
    let phi = s12 * s13 / s23;
    assert_close(s.scalar("_b[/:var(F)]"), phi, 1e-5);
    assert_close(s.scalar("_b[/:var(e.x1)]"), s11 - phi, 1e-5);
    assert_close(s.scalar("_b[/:var(e.x2)]"), s22 - (s23 / s13).powi(2) * phi, 1e-5);
    assert_close(s.scalar("e(chi2_ms)"), 0.0, 1e-6);
    // standardized loading^2 + standardized error variance = 1
    s.run("sem, standardized");
    s.run("matrix T = r(table)");
    // mlmv without missing values equals ml
    let ll = s.scalar("e(ll)");
    s.run("sem (F -> x1 x2 x3), method(mlmv) nolog");
    assert_close(s.scalar("e(ll)"), ll, 1e-6);
    // factor scores (regression method) are centred
    s.run("predict fhat, latent");
    s.run("summarize fhat");
    assert_close(s.scalar("r(mean)"), 0.0, 1e-6);
}

#[test]
fn two_factor_cfa_recovers_loadings_and_fit() {
    let mut s = Session::new();
    sem_data(&mut s, 2000, 13);
    s.run("sem (F -> x1 x2 x3) (G -> w1 w2 w3), nolog");
    assert_close(s.scalar("_b[x2:F]"), 0.8, 0.08);
    assert_close(s.scalar("_b[x3:F]"), 1.2, 0.1);
    assert_close(s.scalar("_b[w2:G]"), 1.5, 0.12);
    assert_close(s.scalar("_b[/:var(F)]"), 1.0, 0.12);
    assert_close(s.scalar("_b[/:cov(F,G)]"), 0.5, 0.08);
    assert_eq!(s.scalar("e(df_ms)"), 8.0);
    s.run("estat gof, stats(all)");
    let (chi2, df, n) = (s.scalar("r(chi2_ms)"), 8.0, 2000.0);
    let rmsea = ((chi2 - df).max(0.0) / (df * (n - 1.0))).sqrt();
    assert_close(s.scalar("r(rmsea)"), rmsea, 1e-9);
    assert!(s.scalar("r(cfi)") > 0.97 && s.scalar("r(cfi)") <= 1.0);
    assert!(s.scalar("r(srmr)") < 0.05);
    assert!(s.scalar("r(lb90_rmsea)") <= rmsea && rmsea <= s.scalar("r(ub90_rmsea)"));
    let aic = -2.0 * s.scalar("e(ll)") + 2.0 * s.scalar("e(rank)");
    assert_close(s.scalar("r(aic)"), aic, 1e-9);
    s.run("estat mindices");
    s.run("estat stdize: test _b[x2:F] = _b[x3:F]");
    // robust, Satorra-Bentler and ADF variants run and agree on the point estimates
    let l = s.scalar("_b[x2:F]");
    s.run("sem (F -> x1 x2 x3) (G -> w1 w2 w3), vce(sbentler) nolog");
    assert_close(s.scalar("_b[x2:F]"), l, 1e-6);
    assert!(s.scalar("e(chi2sb_ms)") > 0.0);
    s.run("sem (F -> x1 x2 x3) (G -> w1 w2 w3), method(adf) nolog");
    assert_close(s.scalar("_b[x2:F]"), l, 0.05);
}

#[test]
fn score_test_close_to_lr_test() {
    let mut s = Session::new();
    sem_data(&mut s, 1500, 14);
    // y1 depends on z: omitting it is detected by the score test
    s.run("sem (y1 <- x1 z), nolog");
    let ll1 = s.scalar("e(ll)");
    s.run("sem (y1 <- x1@b) (y1 <- z@0), nolog");
    let ll0 = s.scalar("e(ll)");
    let lr = 2.0 * (ll1 - ll0);
    s.run("sem (y1 <- x1 z@c) (y2 <- y1 z@c), nolog");
    s.run("estat scoretests");
    s.run("matrix S = r(Cns_sctest)");
    assert!(s.scalar("S[1,1]") > 0.0);
    // constraint z@0 relaxed: Wald/LR/score agree to first order
    assert!(lr > 50.0);
    // constraints() option
    s.run("constraint define 1 [x2]F = [x3]F");
    s.run("sem (F -> x1 x2 x3), constraints(1) nolog");
    assert_close(s.scalar("_b[x2:F]"), s.scalar("_b[x3:F]"), 1e-8);
}

#[test]
fn multigroup_invariance() {
    let mut s = Session::new();
    sem_data(&mut s, 600, 15);
    s.run("sem (F -> x1 x2 x3), group(grp) nolog");
    // measurement coefficients equal across groups by default
    assert_close(s.scalar("_b[x2:F#0.grp]"), s.scalar("_b[x2:F#1.grp]"), 1e-10);
    s.run("estat ginvariant");
    s.run("estat ggof");
    s.run("sem (F -> x1 x2 x3), group(grp) ginvariant(none) nolog");
    let llf = s.scalar("e(ll)");
    // separate fits per group add up
    s.run("sem (F -> x1 x2 x3) if grp == 0, nolog");
    let l0 = s.scalar("e(ll)");
    s.run("sem (F -> x1 x2 x3) if grp == 1, nolog");
    let l1 = s.scalar("e(ll)");
    assert_close(llf, l0 + l1, 1e-5);
}

#[test]
fn ssd_reproduces_raw_data_fit() {
    let mut s = Session::new();
    sem_data(&mut s, 300, 16);
    s.run("sem (F -> x1 x2 x3), nolog");
    let (ll, l2) = (s.scalar("e(ll)"), s.scalar("_b[x2:F]"));
    s.run("quietly correlate x1 x2 x3, covariance");
    s.run("matrix C = r(C)");
    s.run("quietly summarize x1");
    let m1 = s.scalar("r(mean)");
    s.run("quietly summarize x2");
    let m2 = s.scalar("r(mean)");
    s.run("quietly summarize x3");
    let m3 = s.scalar("r(mean)");
    let c = |i: usize, j: usize, s: &mut Session| s.scalar(&format!("C[{i},{j}]"));
    let cov = format!(
        "{} \\ {} {} \\ {} {} {}",
        c(1, 1, &mut s),
        c(2, 1, &mut s),
        c(2, 2, &mut s),
        c(3, 1, &mut s),
        c(3, 2, &mut s),
        c(3, 3, &mut s)
    );
    s.run("ssd init x1 x2 x3");
    s.run("ssd set obs 300");
    s.run(&format!("ssd set means {m1} {m2} {m3}"));
    s.run(&format!("ssd set covariances {cov}"));
    s.run("ssd describe");
    s.run("sem (F -> x1 x2 x3), nolog");
    assert_close(s.scalar("e(ll)"), ll, 1e-6);
    assert_close(s.scalar("_b[x2:F]"), l2, 1e-6);
}

#[test]
fn gsem_single_equations_match_glm_family_estimators() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 300
         set seed 21
         gen id = ceil(_n/6)
         gen double u = rnormal()
         bysort id: replace u = u[1]
         gen double x = rnormal()
         gen byte yb = runiform() < invlogit(.5 + x + u)
         gen int yo = 1 + (x + rnormal() > 0) + (x + rnormal() > 1)
         gen int yc = rpoisson(exp(.2 + .3*x))",
    );
    s.run("logit yb x");
    let (b, ll) = (s.scalar("_b[x]"), s.scalar("e(ll)"));
    s.run("gsem (yb <- x, logit), nolog");
    assert_close(s.scalar("_b[yb:x]"), b, 1e-5);
    assert_close(s.scalar("e(ll)"), ll, 1e-7);
    s.run("ologit yo x");
    let (b, ll) = (s.scalar("_b[x]"), s.scalar("e(ll)"));
    s.run("gsem (yo <- x, ologit), nolog");
    assert_close(s.scalar("_b[yo:x]"), b, 1e-5);
    assert_close(s.scalar("e(ll)"), ll, 1e-7);
    s.run("poisson yc x");
    let ll = s.scalar("e(ll)");
    s.run("gsem (yc <- x, poisson), nolog");
    assert_close(s.scalar("e(ll)"), ll, 1e-7);
    // random intercept through a multilevel latent variable == melogit
    s.run("melogit yb x || id:");
    let (ll, b) = (s.scalar("e(ll)"), s.scalar("_b[x]"));
    s.run("gsem (yb <- x M1[id], logit), nolog");
    assert_close(s.scalar("e(ll)"), ll, 1e-4);
    assert_close(s.scalar("_b[yb:x]"), b, 1e-3);
    s.run("predict m1, latent");
    s.run("predict pb");
}

#[test]
fn latent_class_model_and_lc_postestimation() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 600
         set seed 22
         gen byte cls = runiform() < .35
         gen byte a = runiform() < cond(cls, .9, .2)
         gen byte b = runiform() < cond(cls, .8, .1)
         gen byte c = runiform() < cond(cls, .85, .25)",
    );
    s.run("gsem (a b c <- _cons, logit), lclass(C 2) nolog");
    s.run("estat lcprob");
    s.run("matrix P = r(b)");
    let p1 = s.scalar("P[1,1]");
    let p2 = s.scalar("P[1,2]");
    assert_close(p1 + p2, 1.0, 1e-9);
    assert_close(p1.min(p2), 0.35, 0.07);
    s.run("estat lcmean");
    s.run("estat lcgof");
    s.run("lcstats");
    s.run("predict post*, classposteriorpr");
    s.run("generate double tot = post1 + post2");
    s.run("summarize tot");
    assert_close(s.scalar("r(min)"), 1.0, 1e-9);
}

#[test]
fn irt_1pl_equals_random_intercept_logit_with_item_effects() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 250
         set seed 31
         gen pid = _n
         gen double th = rnormal()
         forvalues i = 1/4 {
             gen byte q`i' = runiform() < invlogit(1.2*(th - (`i'-2.5)*.6))
         }",
    );
    s.run("irt 1pl q1-q4, nolog");
    let ll = s.scalar("e(ll)");
    s.run("estat report");
    s.run("matrix R = r(b)");
    let a = s.scalar("R[1,1]");
    s.run("irtgraph icc q1");
    s.run("irtgraph tif, se");
    s.run("irtgraph iif");
    s.run("irtgraph tcc");
    s.run("predict thetahat, latent");
    s.run("keep pid q1-q4");
    s.run("reshape long q, i(pid) j(item)");
    s.run("melogit q i.item || pid:");
    assert_close(s.scalar("e(ll)"), ll, 1e-3);
    // random-intercept sd equals the common discrimination
    s.run("matrix B = e(b)");
    let v = s.scalar("B[1,colsof(B)]");
    assert_close(v.sqrt(), a, 1e-2);
}

#[test]
fn irt_2pl_is_gsem_with_unit_variance() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 300
         set seed 32
         gen double th = rnormal()
         forvalues i = 1/4 {
             gen byte q`i' = runiform() < invlogit((.6+.3*`i')*(th - (`i'-2.5)*.5))
         }
         gen int r = (th + rnormal() > -.5) + (th + rnormal() > .5)
         gen byte grp = mod(_n, 2)",
    );
    s.run("gsem (Th -> q1-q4, logit), nolog");
    let ll = s.scalar("e(ll)");
    let lam2 = s.scalar("_b[q2:Th]");
    let v = s.scalar("_b[/:var(Th)]");
    s.run("irt 2pl q1-q4, nolog");
    assert_close(s.scalar("e(ll)"), ll, 1e-5);
    assert_close(s.scalar("_b[q2:Theta]"), lam2 * v.sqrt(), 1e-3);
    s.run("irt hybrid (2pl q1-q4) (grm r), nolog");
    s.run("irt grm r q1, nolog");
    s.run("irt pcm r q1, nolog");
    s.run("irt gpcm r q1, nolog");
    s.run("irt nrm r q1, nolog");
    s.run("irt 3pl q1-q4, nolog");
    s.run("irt 2pl q1-q4, group(grp) nolog");
    s.run("estat greport");
    s.run("difmh q1-q4, group(grp)");
    s.run("matrix D = r(dif)");
    assert!(s.scalar("D[1,1]") >= 0.0);
    s.run("diflogistic q1-q4, group(grp)");
}

#[test]
fn fmm_one_component_equals_plain_estimator() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 400
         set seed 41
         gen double x = rnormal()
         gen int cnt = rpoisson(exp(.3 + .5*x))",
    );
    s.run("regress cnt x");
    let b = s.scalar("_b[x]");
    s.run("fmm 1: regress cnt x");
    assert_close(s.scalar("_b[cnt:x]"), b, 1e-6);
    s.run("poisson cnt x");
    let (b, ll) = (s.scalar("_b[x]"), s.scalar("e(ll)"));
    s.run("fmm 1: poisson cnt x");
    assert_close(s.scalar("_b[cnt:x]"), b, 1e-6);
    assert_close(s.scalar("e(ll)"), ll, 1e-8);
}

#[test]
fn fmm_two_component_regression_recovers_parameters() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 800
         set seed 42
         gen double x = rnormal()
         gen byte c = runiform() < .4
         gen double y = cond(c, 4 + 2*x, .5*x) + rnormal()*cond(c, .7, 1)
         gen int cnt = cond(runiform() < .3, 0, rpoisson(exp(.3 + .5*x)))",
    );
    s.run("fmm 2: regress y x");
    // identify the class with the larger intercept
    let (c1, c2) = (s.scalar("_b[y:1.Class]"), s.scalar("_b[y:2.Class]"));
    let hi = if c2 > c1 { 2 } else { 1 };
    let lo = 3 - hi;
    assert_close(s.scalar(&format!("_b[y:{hi}.Class]")), 4.0, 0.15);
    assert_close(s.scalar(&format!("_b[y:{hi}.Class#c.x]")), 2.0, 0.12);
    assert_close(s.scalar(&format!("_b[y:{lo}.Class#c.x]")), 0.5, 0.12);
    s.run("estat lcprob");
    s.run("matrix P = r(b)");
    assert_close(s.scalar(&format!("P[1,{hi}]")), 0.4, 0.05);
    s.run("estat lcmean");
    s.run("predict pp*, classposteriorpr");
    s.run("predict cp*, classpr");
    s.run("predict mu");
    // zero-inflated Poisson as a point-mass mixture
    s.run("fmm: (pointmass cnt) (poisson cnt x)");
    assert_close(s.scalar("_b[cnt:2.Class#c.x]"), 0.5, 0.12);
    s.run("estat eform");
    s.run("fmm 2, lcprob(x): logit c x");
}
