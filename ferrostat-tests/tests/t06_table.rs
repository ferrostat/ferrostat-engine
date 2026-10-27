//! `table` (current and legacy syntax) and the `total` estimator.

use ferrostat_tests::*;

#[test]
fn table_legacy_replace_matches_summarize() {
    let mut s = Session::with_fixture();
    s.run("quietly summarize x if group == 1");
    let (m1, sd1) = (s.scalar("r(mean)"), s.scalar("r(sd)"));
    s.run("quietly count if group == 1");
    let n1 = s.scalar("r(N)");
    let out = s.run("table group, contents(freq mean x sd x) replace");
    assert!(out.contains("Freq.") && out.contains("mean(x)") && out.contains("sd(x)"));
    assert!(s.has_var("table1") && s.has_var("table3"));
    assert_close(s.nums("table1")[0], n1, 1e-12);
    assert_close(s.nums("table2")[0], m1, 1e-10);
    assert_close(s.nums("table3")[0], sd1, 1e-10);
    assert_close(s.nums("table1").iter().sum(), 12.0, 1e-12);
}

#[test]
fn table_legacy_by_and_totals() {
    let mut s = Session::with_fixture();
    let out = s.run("table d, by(group) c(freq mean x)");
    assert!(out.contains("mean(x)"));
    let out = s.run("table d group, c(freq) row col");
    let tot = out.lines().find(|l| l.trim_start().starts_with("Total")).unwrap();
    assert_eq!(*numbers(tot).last().unwrap(), 12.0);
}

#[test]
fn table_current_syntax() {
    let mut s = Session::with_fixture();
    let out = s.run("table group");
    assert!(out.contains("Frequency"));
    let tot = out.lines().find(|l| l.trim_start().starts_with("Total")).unwrap();
    assert_eq!(numbers(tot), vec![12.0]);
    s.run("quietly summarize x");
    let m = s.scalar("r(mean)");
    let out = s.run("table group, statistic(mean x) statistic(sd x) nformat(%9.4f)");
    let tot = out.lines().find(|l| l.trim_start().starts_with("Total")).unwrap();
    assert_close(numbers(tot)[0], (m * 1e4).round() / 1e4, 1e-9);
    let out = s.run("table group d, nototals");
    assert!(!out.contains("Total"));
    s.run("table (group d) (), statistic(percent)");
}

#[test]
fn total_estimator() {
    let mut s = Session::with_fixture();
    s.run("quietly summarize x");
    let (sum, var, n) = (s.scalar("r(sum)"), s.scalar("r(Var)"), s.scalar("r(N)"));
    s.run("total x");
    assert_close(s.scalar("_b[x]"), sum, 1e-9);
    assert_close(s.scalar("_se[x]"), (n * var).sqrt(), 1e-9);
    assert_close(s.scalar("e(N)"), n, 0.0);
    s.run("total x, over(group)");
    s.run("matrix b = e(b)");
    let k = s.scalar("colsof(b)") as usize;
    assert_eq!(s.scalar("e(N_over)") as usize, k);
    let tot: f64 = (1..=k).map(|j| s.scalar(&format!("b[1,{j}]"))).sum();
    assert_close(tot, sum, 1e-9);
}

#[test]
fn total_fweights_match_expanded_data() {
    let mut s = Session::with_fixture();
    s.run("total x [fw=cnt]");
    let (b, se) = (s.scalar("_b[x]"), s.scalar("_se[x]"));
    s.run("drop if cnt < 1 | missing(cnt)");
    s.run("expand cnt");
    s.run("total x");
    assert_close(s.scalar("_b[x]"), b, 1e-9);
    assert_close(s.scalar("_se[x]"), se, 1e-9);
}
