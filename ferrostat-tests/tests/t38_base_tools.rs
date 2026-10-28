//! Base post-estimation tools, prefixes and small statistics commands:
//! lrtest, nlcom, testnl, predictnl, pwcompare/pwmean, estimates
//! save/use/notes/title, estat classification/gof, lroc, brier,
//! jackknife, statsby, rolling, nestreg, stepwise, linktest, ml, mlexp,
//! nl, nlsur, sktest, ladder, esize, kap, icc, loneway, pcorr, bitest,
//! mkspline, orthog, power and the pk commands.
//!
//! Reference values are either published results reproduced on the same
//! data (the Shrout–Fleiss ratings, sample sizes of standard power
//! problems) or identities with independently computed quantities
//! (least squares, logit, lincom, sd/sqrt(n), ...).

use ferrostat_tests::*;

/// Deterministic data: 200 observations, a binary outcome, a group factor.
fn design(s: &mut Session) {
    s.script(
        "clear
         set obs 200
         gen double x1 = mod(_n*37, 101)/101 - 0.5
         gen double x2 = mod(_n*53, 97)/97
         gen double e = mod(_n*7919, 1009)/1009 - 0.5
         gen double y = 1 + 2*x1 - x2 + e
         gen byte g = mod(_n, 4) + 1
         gen byte d = (x1 + 0.6*x2 + 0.8*e > 0.25)
         gen double t = _n",
    );
}

#[test]
fn lrtest_nlcom_testnl() {
    let mut s = Session::new();
    design(&mut s);
    s.run("quietly logit d x1 x2");
    let ll1 = s.scalar("e(ll)");
    s.run("estimates store full");
    s.run("quietly logit d x1");
    let ll0 = s.scalar("e(ll)");
    let out = s.run("lrtest full .");
    assert!(out.contains("Assumption: . nested within full"), "{out}");
    assert_close(s.scalar("r(chi2)"), 2.0 * (ll1 - ll0), 1e-9);
    assert_eq!(s.scalar("r(df)"), 1.0);
    // nlcom of a linear combination equals lincom
    s.run("quietly regress y x1 x2");
    s.run("lincom x1 - 2*x2");
    let (est, se) = (s.scalar("r(estimate)"), s.scalar("r(se)"));
    s.run("nlcom (d: _b[x1] - 2*_b[x2])");
    s.run("matrix NB = r(b)");
    s.run("matrix NV = r(V)");
    assert_close(s.scalar("NB[1,1]"), est, 1e-9);
    assert_close(s.scalar("sqrt(NV[1,1])"), se, 1e-6);
    // testnl of a linear restriction equals test
    s.run("test x1 = 2*x2");
    let f = s.scalar("r(F)");
    s.run("testnl _b[x1] = 2*_b[x2]");
    assert_close(s.scalar("r(F)"), f, 1e-6);
    // delta method for a ratio
    s.run("nlcom _b[x1]/_b[x2], post");
    let (b1, b2) = (2.0f64, -1.0f64);
    assert!((s.scalar("_b[_nl_1]") - b1 / b2).abs() < 0.2);
    assert_eq!(s.run("display e(cmd)").trim(), "nlcom");
}

#[test]
fn predictnl_matches_predict_and_delta_method() {
    let mut s = Session::new();
    design(&mut s);
    s.run("quietly logit d x1 x2");
    s.run("predictnl p = predict(pr), se(sep)");
    s.run("predict p2");
    s.run("predict s2, stdp");
    s.run("gen double dif = abs(p - p2)");
    s.run("gen double dse = abs(sep - s2*p2*(1-p2))");
    s.run("summarize dif");
    assert!(s.scalar("r(max)") < 1e-6);
    s.run("summarize dse");
    assert!(s.scalar("r(max)") < 1e-5);
}

#[test]
fn pwcompare_and_pwmean() {
    let mut s = Session::new();
    design(&mut s);
    s.run("quietly regress y i.g");
    s.run("lincom 3.g - 2.g");
    let (est, se) = (s.scalar("r(estimate)"), s.scalar("r(se)"));
    let out = s.run("pwcompare g, effects");
    assert!(out.contains("Pairwise comparisons of marginal linear predictions"), "{out}");
    assert!(out.contains("3 vs 2"), "{out}");
    assert_close(s.scalar("el(r(b_vs),1,4)"), est, 1e-9);
    assert_close(s.scalar("sqrt(el(r(V_vs),4,4))"), se, 1e-9);
    // Bonferroni: p-values multiplied by the number of comparisons
    s.run("pwcompare g, effects mcompare(bonferroni)");
    let p_b = s.scalar("el(r(table_vs),4,4)");
    s.run("pwcompare g, effects");
    let p_u = s.scalar("el(r(table_vs),4,4)");
    assert_close(p_b, (6.0 * p_u).min(1.0), 1e-9);
    // Tukey and Scheffe adjusted p-values are larger than unadjusted ones
    s.run("pwcompare g, effects mcompare(tukey)");
    assert!(s.scalar("el(r(table_vs),4,4)") >= p_u);
    s.run("pwcompare g, effects mcompare(scheffe)");
    assert!(s.scalar("el(r(table_vs),4,4)") >= p_u);
    // pwmean gives the same comparisons as regress + pwcompare
    s.run("pwmean y, over(g)");
    assert_close(s.scalar("el(r(b_vs),1,4)"), est, 1e-9);
    assert_close(s.scalar("sqrt(el(r(V_vs),4,4))"), se, 1e-9);
    // margins, pwcompare
    s.run("quietly regress y i.g x1");
    let out = s.run("margins g, pwcompare");
    assert!(out.contains("Pairwise comparisons of"), "{out}");
}

#[test]
fn tukey_critical_value() {
    // studentized range quantile q(0.95; k = 5, df = 64) = 3.97 (tables)
    let mut s = Session::new();
    design(&mut s);
    s.run("quietly regress y i.g");
    s.run("pwcompare g, mcompare(tukey)");
    let crit = s.scalar("el(r(table_vs),8,1)") * 2f64.sqrt();
    // k = 4 groups, df = 196: q(0.95; 4, 196) ≈ 3.66
    assert!((crit - 3.66).abs() < 0.02, "{crit}");
}

#[test]
fn estimates_save_use_notes_title() {
    let mut s = Session::new();
    design(&mut s);
    s.run("quietly regress y x1 x2");
    let b = s.scalar("_b[x1]");
    s.run("estimates title: Linear model");
    s.run("estimates notes: first note");
    let out = s.run("estimates notes");
    assert!(out.contains("1. first note"), "{out}");
    let path = std::env::temp_dir().join("ferrostat_t38_est.ster");
    let p = path.display().to_string();
    s.run(&format!("estimates save \"{p}\", replace"));
    s.run("quietly regress y x2");
    s.run(&format!("estimates use \"{p}\""));
    assert_close(s.scalar("_b[x1]"), b, 1e-12);
    assert_eq!(s.run("display e(_estimates_title)").trim(), "Linear model");
    let out = s.run("estimates describe");
    assert!(out.contains("regress y x1 x2"), "{out}");
    s.run("estimates store m1");
    let out = s.run("estimates dir");
    assert!(out.contains("m1") && out.contains("regress"), "{out}");
    let _ = std::fs::remove_file(path);
}

#[test]
fn estimates_for_runs_on_each_model() {
    let mut s = Session::new();
    design(&mut s);
    s.run("quietly regress y x1");
    s.run("estimates store a");
    s.run("quietly regress y x1 x2");
    s.run("estimates store b");
    let out = s.run("estimates for a b: display e(df_m)");
    assert!(out.contains("Model a") && out.contains("Model b"), "{out}");
}

#[test]
fn binary_postestimation() {
    let mut s = Session::new();
    design(&mut s);
    s.run("quietly logit d x1 x2");
    let out = s.run("estat classification");
    assert!(out.contains("Correctly classified"), "{out}");
    let n = s.scalar("r(N)");
    assert_eq!(n, 200.0);
    // correct classification from predicted probabilities
    s.run("predict pr");
    s.run("gen byte ok = (pr >= .5) == d");
    s.run("summarize ok");
    let m = s.scalar("r(mean)");
    s.run("estat classification");
    assert_close(m * 100.0, s.scalar("r(P_corr)"), 1e-9);
    s.run("estat gof, group(10)");
    assert_eq!(s.scalar("r(df)"), 8.0);
    assert!(s.scalar("r(chi2)") >= 0.0);
    s.run("lroc, nograph");
    let auc = s.scalar("r(area)");
    s.run("roctab d pr");
    assert_close(auc, s.scalar("r(area)"), 1e-9);
    s.run("brier d pr");
    // Yates decomposition of the Brier score
    let lhs = s.scalar("r(brier)");
    let rhs = s.scalar("r(oiv) + r(Var_f) + r(relinla) - r(cov_2f)");
    assert_close(lhs, rhs, 1e-9);
}

#[test]
fn jackknife_mean_equals_classical_se() {
    let mut s = Session::new();
    design(&mut s);
    let out = s.run("jackknife r(mean), rclass: summarize y");
    assert!(out.contains("Jackknife results"), "{out}");
    let se = s.scalar("_se[_jk_1]");
    s.run("quietly summarize y");
    assert_close(se, s.scalar("r(sd)/sqrt(r(N))"), 1e-9);
    s.run("jackknife: regress y x1 x2");
    assert_eq!(s.run("display e(vce)").trim(), "jackknife");
    assert_eq!(s.scalar("e(N_reps)"), 200.0);
}

#[test]
fn statsby_and_rolling() {
    let mut s = Session::new();
    design(&mut s);
    s.run("quietly regress y x1 if g == 2");
    let b2 = s.scalar("_b[x1]");
    s.run("statsby _b, by(g) clear nodots: regress y x1");
    assert_eq!(s.n(), 4);
    assert_close(s.scalar("_b_x1[2]"), b2, 1e-5);
    let mut s = Session::new();
    design(&mut s);
    s.run("tsset t");
    s.run("quietly regress y x1 if t >= 11 & t <= 60");
    let b = s.scalar("_b[x1]");
    s.run("rolling _b, window(50) stepsize(10) clear nodots: regress y x1");
    assert_eq!(s.scalar("start[2]"), 11.0);
    assert_close(s.scalar("_b_x1[2]"), b, 1e-5);
}

#[test]
fn nestreg_stepwise_linktest() {
    let mut s = Session::new();
    design(&mut s);
    s.run("nestreg: regress y (x1) (x2 e)");
    s.run("quietly regress y x1 x2 e");
    let r2 = s.scalar("e(r2)");
    s.run("nestreg: regress y (x1) (x2 e)");
    assert_close(s.scalar("el(r(wald),2,5)"), r2, 1e-9);
    let out = s.run("stepwise, pr(.2): regress d x1 x2 g");
    assert!(out.contains("begin with full model"), "{out}");
    assert!(out.contains("removing g"), "{out}");
    s.run("quietly regress y x1 x2");
    s.run("linktest");
    assert!(s.ds().var_index.contains_key("_hatsq"));
    assert_eq!(s.run("display e(cmd)").trim(), "regress");
}

#[test]
fn ml_lf_and_d0_match_built_in_estimators() {
    let mut s = Session::new();
    design(&mut s);
    s.script(
        "program myols
           args lnf xb lnsigma
           quietly replace `lnf' = ln(normalden($ML_y1, `xb', exp(`lnsigma')))
         end",
    );
    s.run("ml model lf myols (y = x1 x2) /lnsigma");
    s.run("ml maximize");
    let (b1, ll) = (s.scalar("_b[eq1:x1]"), s.scalar("e(ll)"));
    s.run("quietly regress y x1 x2");
    assert_close(b1, s.scalar("_b[x1]"), 1e-6);
    let sig = s.scalar("sqrt(e(rss)/e(N))");
    let n = 200.0f64;
    let ll_ols = -n / 2.0 * (2.0 * std::f64::consts::PI * sig * sig).ln() - n / 2.0;
    assert_close(ll, ll_ols, 1e-8);
    s.script(
        "program mylogit
           args todo b lnf
           tempvar xb
           mleval `xb' = `b', eq(1)
           mlsum `lnf' = cond($ML_y1, ln(invlogit(`xb')), ln(invlogit(-`xb')))
         end",
    );
    s.run("ml model d0 mylogit (d = x1 x2)");
    s.run("ml maximize");
    let (b, se) = (s.scalar("_b[x1]"), s.scalar("_se[x1]"));
    s.run("quietly logit d x1 x2");
    assert_close(b, s.scalar("_b[x1]"), 1e-5);
    assert_close(se, s.scalar("_se[x1]"), 1e-4);
}

#[test]
fn mlexp_nl_nlsur() {
    let mut s = Session::new();
    design(&mut s);
    s.run("mlexp (cond(d, ln(invlogit({xb: x1 x2 _cons})), ln(invlogit(-{xb:}))))");
    let b = s.scalar("_b[xb:x1]");
    s.run("quietly logit d x1 x2");
    assert_close(b, s.scalar("_b[x1]"), 1e-5);
    s.run("quietly regress y x1 x2");
    let (b1, se1, rss) = (s.scalar("_b[x1]"), s.scalar("_se[x1]"), s.scalar("e(rss)"));
    let out = s.run("nl (y = {b0} + {b1}*x1 + {b2}*x2)");
    assert!(out.contains("Parameter b0 is used as a constant term"), "{out}");
    assert_close(s.scalar("_b[b1:_cons]"), b1, 1e-6);
    assert_close(s.scalar("_se[b1:_cons]"), se1, 1e-5);
    assert_close(s.scalar("e(rss)"), rss, 1e-8);
    s.run("nlsur (y = {a0} + {a1}*x1) (x2 = {c0} + {c1}*x1)");
    s.run("quietly regress y x1");
    let a1 = s.scalar("_b[x1]");
    s.run("nlsur (y = {a0} + {a1}*x1) (x2 = {c0} + {c1}*x1)");
    // identical regressors: FGNLS equals equation-by-equation OLS
    assert_close(s.scalar("_b[a1:_cons]"), a1, 1e-6);
}

#[test]
fn normality_and_ladder() {
    let mut s = Session::new();
    design(&mut s);
    s.run("gen double z = exp(y)");
    let out = s.run("sktest z");
    assert!(out.contains("Adj chi2(2)"), "{out}");
    assert!(s.scalar("r(p_skew)") < 0.05);
    let out = s.run("ladder z");
    assert!(out.contains("Square root") && out.contains("1/Cubic"), "{out}");
}

#[test]
fn esize_matches_ttest() {
    let mut s = Session::new();
    design(&mut s);
    s.run("ttest y, by(d)");
    let (n1, n2) = (s.scalar("r(N_1)"), s.scalar("r(N_2)"));
    let diff = s.scalar("r(mu_1) - r(mu_2)");
    s.run("esize twosample y, by(d)");
    let d = s.scalar("r(d)");
    // d * sqrt(n1 n2 / (n1 + n2)) is the t statistic
    s.run("ttest y, by(d)");
    assert_close(d * (n1 * n2 / (n1 + n2)).sqrt(), s.scalar("r(t)"), 1e-9);
    assert!(diff.signum() == d.signum());
    assert!(s.scalar("r(t)").is_finite());
}

#[test]
fn agreement_icc_kappa() {
    let mut s = Session::new();
    // Shrout and Fleiss (1979): 6 targets rated by 4 judges
    s.script(
        "clear
         input target judge rating
         1 1 9
         1 2 2
         1 3 5
         1 4 8
         2 1 6
         2 2 1
         2 3 3
         2 4 2
         3 1 8
         3 2 4
         3 3 6
         3 4 8
         4 1 7
         4 2 1
         4 3 2
         4 4 6
         5 1 10
         5 2 5
         5 3 6
         5 4 9
         6 1 6
         6 2 2
         6 3 4
         6 4 7
         end",
    );
    s.run("icc rating target judge");
    assert_close(s.scalar("r(icc_i)"), 0.2897638, 1e-6);
    assert_close(s.scalar("r(icc_i_lb)"), 0.0187865, 1e-5);
    assert_close(s.scalar("r(icc_i_ub)"), 0.7610844, 1e-5);
    s.run("icc rating target");
    assert_close(s.scalar("r(icc_i)"), 0.1657418, 1e-6);
    s.run("icc rating target judge, consistency");
    assert_close(s.scalar("r(icc_i)"), 0.7148407, 1e-6);
    // two raters: perfect agreement gives kappa 1
    s.script("gen r1 = rating > 5\n gen r2 = rating > 5");
    s.run("kap r1 r2");
    assert_close(s.scalar("r(kappa)"), 1.0, 1e-12);
    s.run("loneway rating target");
    assert!(s.scalar("r(rho)") > 0.0);
}

#[test]
fn pcorr_bitest_mkspline_orthog() {
    let mut s = Session::new();
    design(&mut s);
    s.run("quietly regress y x1 x2");
    let t = s.scalar("_b[x1]/_se[x1]");
    s.run("pcorr y x1 x2");
    let pr = s.scalar("el(r(p_corr),1,1)");
    assert_close(pr, t / (t * t + 197.0).sqrt(), 1e-9);
    s.run("bitesti 10 3 .5");
    assert_close(s.scalar("r(p)"), 0.34375, 1e-9);
    assert_close(s.scalar("r(p_l)"), 0.171875, 1e-9);
    s.run("mkspline a1 0 a2 = x1");
    s.run("gen double chk = abs(a1 + a2 - x1)");
    s.run("summarize chk");
    assert!(s.scalar("r(max)") < 1e-6);
    s.run("mkspline c 4 = x2, cubic");
    assert!(s.ds().var_index.contains_key("c3"));
    s.run("orthog x1 x2, gen(o1 o2)");
    s.run("correlate o1 o2");
    assert!(s.scalar("r(rho)").abs() < 1e-9);
    s.run("cumul x1, gen(cx)");
    s.run("summarize cx");
    assert_eq!(s.scalar("r(max)"), 1.0);
}

#[test]
fn power_sample_sizes() {
    let mut s = Session::new();
    s.run("power onemean 0 0.5");
    assert_eq!(s.scalar("r(N)"), 34.0);
    s.run("power twomeans 0 0.5");
    assert_eq!(s.scalar("r(N)"), 128.0);
    s.run("power twoproportions 0.3 0.5");
    assert_eq!(s.scalar("r(N)"), 186.0);
    s.run("power onemean 0 0.5, n(34)");
    assert!(s.scalar("r(power)") >= 0.8);
    s.run("power onemean 0, n(34)");
    assert_close(s.scalar("r(power)"), 0.8, 1e-6);
}

#[test]
fn pk_commands() {
    let mut s = Session::new();
    s.script(
        "clear
         input id time conc
         1 0 0
         1 1 4
         1 2 6
         1 4 5
         1 8 3
         1 12 2
         1 24 1
         2 0 0
         2 1 3
         2 2 7
         2 4 6
         2 8 4
         2 12 2.5
         2 24 0.8
         end",
    );
    s.run("pkexamine time conc if id == 1");
    // trapezoidal AUC: 2 + 5 + 11 + 16 + 10 + 18 = 62
    assert_close(s.scalar("r(auc)"), 62.0, 1e-12);
    assert_eq!(s.scalar("r(cmax)"), 6.0);
    assert_eq!(s.scalar("r(tomc)"), 2.0);
    s.run("pkcollapse time conc, id(id) stat(auc cmax)");
    assert_eq!(s.n(), 2);
    assert_close(s.scalar("auc_conc[1]"), 62.0, 1e-6);
}

#[test]
fn boxcox_and_smoothers() {
    let mut s = Session::new();
    design(&mut s);
    s.run("gen double z = exp(y/2)");
    s.run("boxcox z x1 x2");
    // theta = 0 test: restricted likelihood at log(z) equals the OLS
    // likelihood of log(z) minus the Jacobian term
    assert!(s.scalar("e(ll)") >= s.scalar("e(ll0)"));
    s.run("lowess y x1, gen(ly) nograph");
    s.run("lpoly y x1, gen(gx gy) n(10) nograph");
    assert_eq!(s.scalar("r(ngrid)"), 10.0);
}
