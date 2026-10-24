//! Hypothesis tests and correlation-based statistics.

use ferrostat_tests::reference as r;
use ferrostat_tests::*;

#[test]
fn ttest_one_sample() {
    let mut s = Session::with_fixture();
    s.run("ttest y == 10");
    assert_close(s.scalar("r(t)"), r::TT1_T, 1e-10);
    assert_close(s.scalar("r(se)"), r::TT1_SE, 1e-10);
    assert_eq!(s.scalar("r(df_t)"), 11.0);
    let p = 2.0 * s.scalar("ttail(11, abs(r(t)))");
    assert_close(s.scalar("r(p)"), p, 1e-12);
}

#[test]
fn ttest_paired() {
    let mut s = Session::with_fixture();
    s.run("ttest y == x");
    assert_close(s.scalar("r(t)"), r::TTP_T, 1e-10);
    assert_close(s.scalar("r(mu_diff)"), r::TTP_MEAN, 1e-10);
}

#[test]
fn ttest_two_groups_equal_and_unequal() {
    let mut s = Session::with_fixture();
    s.run("ttest y, by(d)");
    assert_close(s.scalar("r(t)"), r::TTG_T, 1e-10);
    assert_eq!(s.scalar("r(df_t)"), 10.0);
    s.run("ttest y, by(d) unequal");
    assert_close(s.scalar("r(t)"), r::TTW_T, 1e-10);
    assert_close(s.scalar("r(df_t)"), r::TTW_DF, 1e-8);
}

#[test]
fn ttest_unpaired_variables() {
    let mut s = Session::with_fixture();
    s.run("ttest x == y, unpaired");
    assert_close(s.scalar("r(t)"), r::TTU_T, 1e-10);
}

#[test]
fn tabulate_chi2_stored() {
    let mut s = Session::with_fixture();
    s.run("tabulate group d, chi2");
    assert_close(s.scalar("r(chi2)"), r::CHI2_GD, 1e-12);
    assert_close(s.scalar("r(p)"), (-1.0f64).exp(), 1e-9);
}

#[test]
fn oneway_anova() {
    let mut s = Session::with_fixture();
    s.run("oneway y group");
    assert_close(s.scalar("r(F)"), r::ANOVA_F, 1e-9);
}

#[test]
fn spearman_rank_correlation() {
    let mut s = Session::with_fixture();
    s.run("spearman x z");
    assert_close(s.scalar("r(rho)"), r::SPEARMAN_XZ, 1e-12);
}

#[test]
fn correlate_and_covariance() {
    let mut s = Session::with_fixture();
    s.run("correlate x y");
    assert_close(s.scalar("r(rho)"), r::CORR_XY, 1e-12);
    s.run("correlate x y, covariance");
    assert_close(s.scalar("r(cov_12)"), r::COV_XY, 1e-12);
}

#[test]
fn ranksum_and_signrank_run() {
    let mut s = Session::with_fixture();
    s.run("ranksum y, by(d)");
    assert!(s.scalar("r(p)") > 0.0 && s.scalar("r(p)") < 1.0);
    s.run("signrank y = x");
    assert_eq!(s.scalar("r(N)"), 12.0);
}

#[test]
fn kwallis_and_sdtest_run() {
    let mut s = Session::with_fixture();
    s.run("kwallis y, by(group)");
    assert_eq!(s.scalar("r(df)"), 2.0);
    s.run("sdtest y, by(d)");
    assert!(s.scalar("r(F)") > 0.0);
}

#[test]
fn prtest_one_sample() {
    let mut s = Session::with_fixture();
    s.run("prtest d == 0.5");
    assert_close(s.scalar("r(z)"), 0.0, 1e-12);
}

#[test]
fn swilk_runs() {
    let mut s = Session::with_fixture();
    s.run("swilk y");
    let w = s.scalar("r(W)");
    assert!(w > 0.8 && w <= 1.0, "{w}");
}

#[test]
fn summarize_detail_percentiles() {
    let mut s = Session::with_fixture();
    s.run("summarize y, detail");
    assert_close(s.scalar("r(p50)"), r::Y_P50, 1e-12);
    assert_close(s.scalar("r(skewness)"), r::Y_SKEW, 1e-9);
}
