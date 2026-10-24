//! Estimation commands and post-estimation, checked against the
//! independent reference implementation in reference/reference.py.

use ferrostat_tests::reference as r;
use ferrostat_tests::*;

const TOL: f64 = 1e-8;

#[test]
fn regress_coefficients_and_standard_errors() {
    let mut s = Session::with_fixture();
    s.run("regress y x z");
    for (j, name) in ["x", "z", "_cons"].iter().enumerate() {
        assert_close(s.scalar(&format!("_b[{name}]")), r::REG_B[j], TOL);
        assert_close(s.scalar(&format!("_se[{name}]")), r::REG_SE[j], 1e-7);
    }
}

#[test]
fn regress_model_statistics() {
    let mut s = Session::with_fixture();
    s.run("regress y x z");
    assert_eq!(s.scalar("e(N)"), 12.0);
    assert_eq!(s.scalar("e(df_m)"), 2.0);
    assert_eq!(s.scalar("e(df_r)"), 9.0);
    assert_close(s.scalar("e(r2)"), r::REG_R2, TOL);
    assert_close(s.scalar("e(r2_a)"), r::REG_ADJ_R2, TOL);
    assert_close(s.scalar("e(rmse)"), r::REG_RMSE, TOL);
    assert_close(s.scalar("e(F)"), r::REG_F, 1e-6);
    assert_close(s.scalar("e(rss)"), r::REG_RSS, TOL);
    assert_close(s.scalar("e(mss)"), r::REG_MSS, TOL);
}

#[test]
fn regress_output_table() {
    let mut s = Session::with_fixture();
    let out = s.run("regress y x z");
    let row = row(&out, "x").expect("x row");
    assert_close(row[0], r::REG_B[0], 1e-6);
    assert_close(row[1], r::REG_SE[0], 1e-6);
    assert!(out.contains("Number of obs") && out.contains("R-squared"));
}

#[test]
fn regress_robust_hc1() {
    let mut s = Session::with_fixture();
    s.run("regress y x z, vce(robust)");
    for (j, name) in ["x", "z", "_cons"].iter().enumerate() {
        assert_close(s.scalar(&format!("_se[{name}]")), r::REG_RSE[j], 1e-7);
    }
    assert_eq!(s.run("display e(vce)").trim(), "robust");
}

#[test]
fn regress_noconstant() {
    let mut s = Session::with_fixture();
    s.run("regress y x, noconstant");
    assert_close(s.scalar("_b[x]"), r::REGNC_B[0], TOL);
    assert_close(s.scalar("_se[x]"), r::REGNC_SE[0], 1e-7);
}

#[test]
fn regress_drops_missing_casewise() {
    let mut s = Session::with_missing_fixture();
    // y missing in obs 12 and z missing in obs 1
    s.run("regress y x z");
    assert_eq!(s.scalar("e(N)"), 10.0);
}

#[test]
fn regress_if_restriction_matches_subset() {
    let mut s = Session::with_fixture();
    s.run("regress y x z in 1/11");
    for (j, name) in ["x", "z", "_cons"].iter().enumerate() {
        assert_close(s.scalar(&format!("_b[{name}]")), r::REG11_B[j], 1e-7);
    }
}

#[test]
fn regress_analytic_weights() {
    let mut s = Session::with_fixture();
    s.run("regress y x [aw=z]");
    assert_close(s.scalar("_b[x]"), r::REGAW_B[0], 1e-8);
    assert_close(s.scalar("_b[_cons]"), r::REGAW_B[1], 1e-8);
}

#[test]
fn regress_factor_variables() {
    let mut s = Session::with_fixture();
    s.run("regress y i.group");
    // with a full set of group dummies the fit reproduces the group means
    assert_close(s.scalar("_b[_cons]"), r::GROUP_MEAN_Y[0], 1e-9);
    assert_close(s.scalar("_b[2.group]"), r::GROUP_MEAN_Y[1] - r::GROUP_MEAN_Y[0], 1e-9);
    assert_close(s.scalar("_b[3.group]"), r::GROUP_MEAN_Y[2] - r::GROUP_MEAN_Y[0], 1e-9);
    assert_eq!(s.scalar("_b[1.group]"), 0.0);
}

#[test]
fn regress_collinear_variable_is_omitted() {
    let mut s = Session::with_fixture();
    s.run("generate double x2 = 2*x");
    let out = s.run("regress y x x2");
    assert!(out.contains("omitted"), "{out}");
    assert_close(s.scalar("_b[x]"), r::REG1_B[0], 1e-8);
}

#[test]
fn regress_cluster_runs_and_counts_clusters() {
    let mut s = Session::with_fixture();
    s.run("regress y x, vce(cluster group)");
    assert_eq!(s.scalar("e(N_clust)"), 3.0);
    assert_eq!(s.scalar("e(df_r)"), 2.0);
}

#[test]
fn time_series_operators_in_regressors() {
    let mut s = Session::with_fixture();
    s.run("tsset id");
    s.run("regress y L.x");
    assert_eq!(s.scalar("e(N)"), 11.0);
}

#[test]
fn predict_xb_and_residuals() {
    let mut s = Session::with_fixture();
    s.run("regress y x z");
    s.run("predict double yhat");
    s.run("predict double e, residuals");
    assert_all_close(&s.nums("yhat"), &r::REG_FIT, 1e-9);
    assert_all_close(&s.nums("e"), &r::REG_RESID, 1e-9);
}

#[test]
fn predict_requires_estimates() {
    let mut s = Session::with_fixture();
    s.fails("predict p");
}

#[test]
fn test_joint_and_single_hypotheses() {
    let mut s = Session::with_fixture();
    s.run("regress y x z");
    s.run("test z");
    let t = r::REG_B[1] / r::REG_SE[1];
    assert_close(s.scalar("r(F)"), t * t, 1e-6);
    s.run("test x = 2");
    let t2 = (r::REG_B[0] - 2.0) / r::REG_SE[0];
    assert_close(s.scalar("r(F)"), t2 * t2, 1e-6);
    s.run("test x z");
    assert_close(s.scalar("r(F)"), r::REG_F, 1e-6);
}

#[test]
fn lincom_combination() {
    let mut s = Session::with_fixture();
    s.run("regress y x z");
    s.run("lincom x + z");
    assert_close(s.scalar("r(estimate)"), r::REG_B[0] + r::REG_B[1], 1e-9);
}

#[test]
fn logit_matches_reference() {
    let mut s = Session::with_fixture();
    s.run("logit d x");
    assert_close(s.scalar("_b[x]"), r::LOGIT_B[0], 1e-6);
    assert_close(s.scalar("_b[_cons]"), r::LOGIT_B[1], 1e-6);
    assert_close(s.scalar("_se[x]"), r::LOGIT_SE[0], 1e-5);
    assert_close(s.scalar("e(ll)"), r::LOGIT_LL, 1e-8);
    assert_close(s.scalar("e(ll_0)"), r::LOGIT_LL0, 1e-8);
    assert_close(s.scalar("e(chi2)"), r::LOGIT_CHI2, 1e-6);
    assert_close(s.scalar("e(r2_p)"), r::LOGIT_R2P, 1e-7);
}

#[test]
fn logit_predict_probability() {
    let mut s = Session::with_fixture();
    s.run("logit d x");
    s.run("predict double p");
    let p = s.nums("p");
    let expected = 1.0 / (1.0 + (-(r::LOGIT_B[1] + r::LOGIT_B[0] * 5.0)).exp());
    assert_close(p[4], expected, 1e-6);
}

#[test]
fn logistic_reports_odds_ratios() {
    let mut s = Session::with_fixture();
    let out = s.run("logistic d x");
    assert!(out.contains("Odds ratio"), "{out}");
    let v = row(&out, "x").unwrap();
    assert_close(v[0], r::LOGIT_B[0].exp(), 1e-5);
}

#[test]
fn probit_matches_reference() {
    let mut s = Session::with_fixture();
    s.run("probit d x");
    assert_close(s.scalar("_b[x]"), r::PROBIT_B[0], 1e-6);
    assert_close(s.scalar("_se[x]"), r::PROBIT_SE[0], 1e-5);
    assert_close(s.scalar("e(ll)"), r::PROBIT_LL, 1e-8);
}

#[test]
fn poisson_matches_reference() {
    let mut s = Session::with_fixture();
    s.run("poisson cnt x");
    assert_close(s.scalar("_b[x]"), r::POIS_B[0], 1e-6);
    assert_close(s.scalar("_b[_cons]"), r::POIS_B[1], 1e-6);
    assert_close(s.scalar("_se[x]"), r::POIS_SE[0], 1e-5);
    assert_close(s.scalar("e(ll)"), r::POIS_LL, 1e-8);
    assert_close(s.scalar("e(chi2)"), r::POIS_CHI2, 1e-6);
}

#[test]
fn anova_one_way() {
    let mut s = Session::with_fixture();
    s.run("anova y group");
    assert_close(s.scalar("e(F)"), r::ANOVA_F, 1e-8);
    assert_close(s.scalar("e(mss)"), r::ANOVA_SSB, 1e-8);
    assert_close(s.scalar("e(rss)"), r::ANOVA_SSW, 1e-8);
    assert_close(s.scalar("e(ss_1)"), r::ANOVA_SSB, 1e-8);
}

#[test]
fn xtreg_fixed_effects() {
    let mut s = Session::with_fixture();
    s.run("xtset group");
    s.run("xtreg y x, fe");
    assert_close(s.scalar("_b[x]"), r::FE_B, 1e-9);
    assert_close(s.scalar("_se[x]"), r::FE_SE, 1e-8);
    assert_close(s.scalar("_b[_cons]"), r::FE_CONS, 1e-8);
    assert_eq!(s.scalar("e(N_g)"), 3.0);
}

#[test]
fn xtreg_random_effects_runs() {
    let mut s = Session::with_fixture();
    s.run("xtset group");
    s.run("xtreg y x, re");
    assert!(s.scalar("_b[x]") > 1.9 && s.scalar("_b[x]") < 2.1);
}

#[test]
fn ivregress_reduces_to_ols_with_exogenous_instrument() {
    let mut s = Session::with_fixture();
    // instrumenting x by itself reproduces OLS
    s.run("ivregress 2sls y (x = x)");
    assert_close(s.scalar("_b[x]"), r::REG1_B[0], 1e-8);
}

#[test]
fn estimates_store_restore_table() {
    let mut s = Session::with_fixture();
    s.run("regress y x");
    s.run("estimates store m1");
    s.run("regress y x z");
    s.run("estimates store m2");
    s.run("estimates restore m1");
    assert_close(s.scalar("_b[x]"), r::REG1_B[0], 1e-9);
    let out = s.run("estimates table m1 m2, se stats(N r2)");
    assert!(out.contains("m1") && out.contains("m2"), "{out}");
}

#[test]
fn margins_linear_model() {
    let mut s = Session::with_fixture();
    s.run("regress y x z");
    s.run("margins");
    assert_close(s.scalar("el(r(b), 1, 1)"), r::Y_MEAN, 1e-8);
    s.run("margins, dydx(x)");
    assert_close(s.scalar("el(r(b), 1, 1)"), r::REG_B[0], 1e-5);
}

#[test]
fn estat_ic() {
    let mut s = Session::with_fixture();
    s.run("logit d x");
    s.run("estat ic");
    assert_close(s.scalar("el(r(S), 1, 5)"), -2.0 * r::LOGIT_LL + 4.0, 1e-8);
}

#[test]
fn estat_firststage_after_ivregress() {
    // used by test-andrea/Spatial.do: matrix F = r(singleresults); F[1,4]
    let mut s = Session::with_fixture();
    s.run("ivregress 2sls y z (x = cnt)");
    s.run("estat firststage");
    // first stage: x on cnt and z; F of the excluded instrument equals t^2
    s.run("matrix F = r(singleresults)");
    let f = s.scalar("F[1,4]");
    s.run("regress x cnt z");
    let t = s.scalar("_b[cnt] / _se[cnt]");
    assert_close(f, t * t, 1e-8);
    assert!(s.scalar("F[1,1]") > 0.0 && s.scalar("F[1,1]") <= 1.0);
}

/// Small balanced panel generated deterministically (no RNG dependence).
fn ardl_panel(s: &mut Session) {
    s.run("clear");
    s.run("set obs 120");
    s.run("generate id = ceil(_n / 30)");
    s.run("bysort id: generate t = _n");
    s.run("generate double x = sin(_n * 0.7) * 3 + id");
    s.run("generate double y = 0");
    s.run("xtset id t");
    // y_t = 0.5 y_{t-1} + 0.4 x_t + id-specific shock pattern
    s.run("by id: replace y = 0.5 * y[_n-1] + 0.4 * x + 0.1 * cos(_n * 1.3 + id) if _n > 1");
}

#[test]
fn xtpmg_mean_group_averages_panel_estimates() {
    // used by test-andrea/Spatial.do Task 6
    let mut s = Session::new();
    ardl_panel(&mut s);
    s.run("xtpmg d.y d.x, lr(l.y x) ec(ec) mg replace");
    let mg = s.scalar("_b[ec:x]");
    // panel-by-panel: theta_i = -b_x / b_ly from D.y on L.y x D.x
    let mut thetas = Vec::new();
    for i in 1..=4 {
        s.run(&format!("quietly regress D.y L.y x D.x if id == {i}"));
        thetas.push(-s.scalar("_b[x]") / s.scalar("_b[L.y]"));
    }
    assert_close(mg, thetas.iter().sum::<f64>() / 4.0, 1e-9);
    assert!(s.has_var("ec"));
}

#[test]
fn xtpmg_pooled_mean_group_and_hausman() {
    let mut s = Session::new();
    ardl_panel(&mut s);
    s.run("xtpmg d.y d.x, lr(l.y x) ec(ec) pmg replace");
    s.run("estimates store PMG");
    let ll = s.scalar("e(ll)");
    assert!(ll.is_finite());
    // the long-run coefficient is close to the true value 0.4 / (1 - 0.5) = 0.8
    assert!((s.scalar("_b[ec:x]") - 0.8).abs() < 0.05, "{}", s.scalar("_b[ec:x]"));
    s.run("xtpmg d.y d.x, lr(l.y x) ec(ec) mg replace");
    s.run("estimates store MG");
    s.run("hausman MG PMG");
    assert_eq!(s.scalar("r(df)"), 1.0);
}

fn iv_panel(s: &mut Session) {
    s.run("clear");
    s.run("set obs 200");
    s.run("generate id = ceil(_n / 10)");
    s.run("bysort id: generate t = _n");
    s.run("xtset id t");
    s.run("generate double z1 = sin(_n * 0.37)");
    s.run("generate double z2 = cos(_n * 1.91)");
    s.run("generate double e = sin(_n * 2.3) * 0.5");
    s.run("generate double x = z1 + 0.5 * z2 + e + id / 10");
    s.run("generate double y = 2 * x + e + id");
}

#[test]
fn xtivreg2_matches_xtivreg_and_reports_diagnostics() {
    // test-andrea/Spatial.do Task 5 uses xtivreg2 ..., fe cluster() endog()
    let mut s = Session::new();
    iv_panel(&mut s);
    s.run("xtivreg y (x = z1 z2), fe");
    let b = s.scalar("_b[x]");
    let se = s.scalar("_se[x]");
    s.run("xtivreg2 y (x = z1 z2), fe endog(x)");
    assert_close(s.scalar("_b[x]"), b, 1e-10);
    assert_close(s.scalar("_se[x]"), se, 1e-10);
    assert!(s.scalar("e(j)") >= 0.0);
    assert_eq!(s.scalar("e(jdf)"), 1.0);
    assert!(s.scalar("e(estat)") >= 0.0);
    // clustering changes only the standard errors
    s.run("xtivreg2 y (x = z1 z2), fe cluster(id)");
    assert_close(s.scalar("_b[x]"), b, 1e-10);
    assert_eq!(s.scalar("e(N_clust)"), 20.0);
    // exactly identified: J = 0
    s.run("xtivreg2 y (x = z1), fe");
    assert_eq!(s.scalar("e(j)"), 0.0);
}
