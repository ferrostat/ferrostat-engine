//! Descriptive statistics and listing commands.

use ferrostat_tests::reference as r;
use ferrostat_tests::*;

// ---- summarize ----------------------------------------------------------------

#[test]
fn summarize_basic_statistics() {
    let mut s = Session::with_fixture();
    let out = s.run("summarize y");
    let v = row(&out, "y").expect("row for y");
    assert_eq!(v[0], 12.0);
    assert_close(v[1], r::Y_MEAN, 1e-6);
    assert_close(v[2], r::Y_SD, 1e-6);
    assert_eq!(v[3], 2.1);
    assert_eq!(v[4], 23.8);
}

#[test]
fn summarize_all_numeric_variables_by_default() {
    let mut s = Session::with_fixture();
    let out = s.run("summarize");
    for v in ["id", "group", "x", "y", "z", "d", "cnt"] {
        assert!(row(&out, v).is_some(), "missing row {v} in\n{out}");
    }
    // string variables are listed with zero observations, never with numbers
    assert!(row(&out, "name").map_or(true, |v| v.iter().all(|&n| n == 0.0)));
}

#[test]
fn summarize_skips_missing_values() {
    let mut s = Session::with_missing_fixture();
    let out = s.run("summarize y");
    assert_eq!(row(&out, "y").unwrap()[0], 11.0);
}

#[test]
fn summarize_if_and_in() {
    let mut s = Session::with_fixture();
    let out = s.run("summarize x if group == 2");
    assert_eq!(row(&out, "x").unwrap()[..2], [4.0, 6.5]);
    let out = s.run("summarize x in 1/3");
    assert_eq!(row(&out, "x").unwrap()[..2], [3.0, 2.0]);
}

#[test]
#[ignore = "bug: sd of a single observation is reported as 0; Stata reports missing"]
fn summarize_single_observation_has_no_sd() {
    let mut s = Session::with_fixture();
    let out = s.run("summarize x in 1");
    let v = row(&out, "x").unwrap();
    assert_eq!(v[0], 1.0);
    // Stata shows a missing sd ('.') for N == 1; the number must not be 0
    assert!(v.len() < 5 || v[2] != 0.0, "{out}");
}

#[test]
#[ignore = "missing: stored results r(N), r(mean), r(sd), r(min), r(max), r(sum)"]
fn summarize_stored_results() {
    let mut s = Session::with_fixture();
    s.run("summarize y");
    assert_eq!(s.scalar("r(N)"), 12.0);
    assert_close(s.scalar("r(mean)"), r::Y_MEAN, 1e-12);
    assert_close(s.scalar("r(sd)"), r::Y_SD, 1e-12);
    assert_close(s.scalar("r(Var)"), r::Y_VAR, 1e-12);
    assert_eq!(s.scalar("r(min)"), 2.1);
    assert_eq!(s.scalar("r(max)"), 23.8);
    assert_close(s.scalar("r(sum)"), Y.iter().sum(), 1e-12);
}

#[test]
#[ignore = "missing: summarize, detail (percentiles, skewness, kurtosis)"]
fn summarize_detail() {
    let mut s = Session::with_fixture();
    s.run("summarize y, detail");
    assert_close(s.scalar("r(p50)"), r::Y_P50, 1e-12);
    assert_close(s.scalar("r(p25)"), r::Y_P25, 1e-12);
    assert_close(s.scalar("r(p75)"), r::Y_P75, 1e-12);
    assert_close(s.scalar("r(p10)"), r::Y_P10, 1e-12);
    assert_close(s.scalar("r(p90)"), r::Y_P90, 1e-12);
    assert_close(s.scalar("r(skewness)"), r::Y_SKEW, 1e-9);
    assert_close(s.scalar("r(kurtosis)"), r::Y_KURT, 1e-9);
}

#[test]
#[ignore = "missing: weights in summarize ([aw=], [fw=])"]
fn summarize_with_frequency_weights() {
    let mut s = Session::with_fixture();
    s.run("summarize x [fw=cnt]");
    let sw: f64 = CNT.iter().sum();
    let m: f64 = X.iter().zip(CNT).map(|(x, c)| x * c).sum::<f64>() / sw;
    assert_eq!(s.scalar("r(N)"), sw);
    assert_close(s.scalar("r(mean)"), m, 1e-12);
}

// ---- tabulate -------------------------------------------------------------------

#[test]
fn tabulate_oneway_frequencies() {
    let mut s = Session::with_fixture();
    let out = s.run("tabulate group");
    for g in ["1", "2", "3"] {
        let v = row(&out, g).unwrap_or_else(|| panic!("row {g}\n{out}"));
        assert_eq!(v[0], 4.0);
        assert_close(v[1], 100.0 / 3.0, 1e-2);
    }
    assert_close(row(&out, "3").unwrap()[2], 100.0, 1e-9);
    assert_eq!(row(&out, "Total").unwrap()[0], 12.0);
}

#[test]
fn tabulate_string_variable() {
    let mut s = Session::with_fixture();
    let out = s.run("tabulate name");
    assert_eq!(row(&out, "koi").unwrap()[0], 1.0);
}

#[test]
#[ignore = "bug: tabulate sorts numeric values as formatted strings (50.0000 before 6.0000)"]
fn tabulate_orders_numeric_values_numerically() {
    let mut s = Session::with_fixture();
    s.run("replace z = z * 10 if z > 4");
    let out = s.run("tabulate z");
    let keys: Vec<f64> = out.lines().filter_map(|l| l.split_once('|')).filter_map(|(h, _)| h.trim().parse().ok()).collect();
    assert!(keys.windows(2).all(|w| w[0] < w[1]), "{out}");
}

#[test]
fn tabulate_twoway_with_chi2() {
    let mut s = Session::with_fixture();
    let out = s.run("tabulate group d");
    assert_close(stat(&out, "Pearson chi2(2) =").expect("chi2 line"), r::CHI2_GD, 1e-9);
}

#[test]
#[ignore = "missing: tabulate shows value labels instead of codes"]
fn tabulate_uses_value_labels() {
    let mut s = Session::with_fixture();
    let out = s.run("tabulate d");
    assert!(row(&out, "yes").is_some(), "{out}");
}

#[test]
#[ignore = "missing: tabulate, missing (count missing as a category)"]
fn tabulate_missing_option() {
    let mut s = Session::with_missing_fixture();
    let without = s.run("tabulate y");
    assert_eq!(row(&without, "Total").unwrap()[0], 11.0);
    let with = s.run("tabulate y, missing");
    assert_eq!(row(&with, "Total").unwrap()[0], 12.0);
}

#[test]
#[ignore = "missing: tabulate stored results r(N), r(r), r(c), r(chi2), r(p) and options row/col/exact/lrchi2"]
fn tabulate_stored_results() {
    let mut s = Session::with_fixture();
    s.run("tabulate group d, chi2 row col");
    assert_eq!(s.scalar("r(N)"), 12.0);
    assert_eq!(s.scalar("r(r)"), 3.0);
    assert_eq!(s.scalar("r(c)"), 2.0);
    assert_close(s.scalar("r(chi2)"), r::CHI2_GD, 1e-12);
    assert_close(s.scalar("r(p)"), (-1.0f64).exp(), 1e-9); // chi2(2)=2 => p = e^-1
}

#[test]
#[ignore = "missing: tab1 / tab2 / table / tabstat"]
fn tabstat_by_group() {
    let mut s = Session::with_fixture();
    let out = s.run("tabstat y, by(group) statistics(mean sd n)");
    assert_close(row(&out, "1").unwrap()[0], r::GROUP_MEAN_Y[0], 1e-4);
}

// ---- correlate --------------------------------------------------------------------

#[test]
fn correlate_matrix() {
    let mut s = Session::with_fixture();
    let out = s.run("correlate x y z");
    let ry = row(&out, "y").unwrap();
    assert_close(ry[0], r::CORR_XY, 1e-4);
    assert_close(ry[1], 1.0, 0.0);
    let rz = row(&out, "z").unwrap();
    assert_close(rz[0], r::CORR_XZ, 1e-4);
    assert_close(rz[1], r::CORR_YZ, 1e-4);
    assert_eq!(stat(&out, "obs ="), Some(12.0));
}

#[test]
fn correlate_needs_two_variables() {
    let mut s = Session::with_fixture();
    s.fails("correlate x");
}

#[test]
#[ignore = "bug: correlate must use casewise deletion; it drops missing values per variable and misaligns pairs"]
fn correlate_casewise_deletion() {
    let mut s = Session::with_missing_fixture();
    let out = s.run("correlate x y z");
    assert_eq!(stat(&out, "obs ="), Some(10.0));
}

#[test]
#[ignore = "missing: correlate stored results r(rho), r(N) and covariance option"]
fn correlate_stored_results_and_covariance() {
    let mut s = Session::with_fixture();
    s.run("correlate x y");
    assert_close(s.scalar("r(rho)"), r::CORR_XY, 1e-12);
    s.run("correlate x y, covariance");
    assert_close(s.scalar("r(cov_12)"), r::COV_XY, 1e-12);
}

#[test]
#[ignore = "missing: pwcorr / spearman"]
fn pwcorr_and_spearman() {
    let mut s = Session::with_fixture();
    s.run("spearman x z");
    assert_close(s.scalar("r(rho)"), r::SPEARMAN_XZ, 1e-12);
    s.run("pwcorr x y z, sig");
}

// ---- count / list / describe ------------------------------------------------------

#[test]
fn count_all_if_in() {
    let mut s = Session::with_fixture();
    assert_eq!(s.run("count").trim(), "12");
    assert_eq!(s.run("count if d == 1").trim(), "6");
    assert_eq!(s.run("count in 2/4").trim(), "3");
}

#[test]
#[ignore = "bug: `count if ... in ...` ignores the if condition when in is also given"]
fn count_if_and_in_combined() {
    let mut s = Session::with_fixture();
    assert_eq!(s.run("count if d == 1 in 1/6").trim(), "2");
}

#[test]
fn list_selected_variables_and_rows() {
    let mut s = Session::with_fixture();
    let out = s.run("list name y in 1/2");
    assert!(out.contains("ant") && out.contains("bee"));
    assert!(!out.contains("cat"));
    assert!(out.contains("2.1"));
}

#[test]
fn list_with_if() {
    let mut s = Session::with_fixture();
    let out = s.run("list name if z == 9");
    assert!(out.contains("fox"));
    assert!(!out.contains("ant"));
}

#[test]
fn list_shows_missing_as_dot() {
    let mut s = Session::with_missing_fixture();
    let out = s.run("list y in 12");
    assert!(out.lines().last().unwrap().trim() == ".", "{out}");
}

#[test]
fn describe_lists_variables_and_types() {
    let mut s = Session::with_fixture();
    let out = s.run("describe");
    for (v, ty) in [("id", "int"), ("name", "str4"), ("group", "byte"), ("x", "double")] {
        let line = out.lines().find(|l| l.split_whitespace().next() == Some(v)).unwrap_or_else(|| panic!("{v}\n{out}"));
        assert!(line.contains(ty), "{line}");
    }
    assert!(out.contains("Binary outcome"));
}

#[test]
#[ignore = "missing: describe header (obs, vars, data label) and r(N), r(k)"]
fn describe_stored_results() {
    let mut s = Session::with_fixture();
    s.run("describe");
    assert_eq!(s.scalar("r(N)"), 12.0);
    assert_eq!(s.scalar("r(k)"), 8.0);
}

#[test]
#[ignore = "missing: codebook / inspect / misstable"]
fn codebook() {
    let mut s = Session::with_missing_fixture();
    let out = s.run("misstable summarize");
    assert!(out.contains('y'));
    s.run("codebook group");
}

// ---- kernel density ---------------------------------------------------------------

#[test]
fn kdensity_reports_bandwidth_and_density() {
    let mut s = Session::with_fixture();
    let out = s.run("kdensity x, kernel(gaussian)");
    assert_close(stat(&out, "Bandwidth:").unwrap(), r::KD_H, 1e-3);
    let at = out.lines().map(numbers).find(|v| v.len() == 2 && (v[0] - 6.5).abs() < 1e-9).expect("grid point 6.5");
    assert_close(at[1], r::KD_AT_6_5, 1e-3);
}

#[test]
#[ignore = "bug: kdensity uses the Gaussian kernel and iqr/1.34 by default; Stata defaults to Epanechnikov and iqr/1.349"]
fn kdensity_default_kernel_is_epanechnikov() {
    let mut s = Session::with_fixture();
    let out = s.run("kdensity x");
    assert!(out.to_lowercase().contains("epanechnikov"), "{out}");
}

#[test]
#[ignore = "missing: kdensity options at(), n(), bwidth(), generate()"]
fn kdensity_generate() {
    let mut s = Session::with_fixture();
    s.run("kdensity x, kernel(gaussian) bwidth(2) n(12) generate(gx gd) nograph");
    assert_eq!(s.nums("gx").len(), 12);
}
