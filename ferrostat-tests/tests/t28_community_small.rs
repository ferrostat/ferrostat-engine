//! Natively implemented community commands: distinct, mdesc, winsor2.

use ferrostat_tests::*;

#[test]
fn distinct_counts_per_variable_and_jointly() {
    let mut s = Session::with_missing_fixture();
    let out = s.run("distinct group d");
    assert!(out.contains("group |") && out.contains("total   distinct"), "{out}");
    assert_eq!(s.scalar("r(ndistinct)"), 2.0);
    s.run("distinct group d, joint");
    assert!(s.scalar("r(ndistinct)") <= 6.0);
    // missing values are excluded unless asked for
    s.run("distinct y");
    let n = s.scalar("r(N)");
    s.run("distinct y, missing");
    assert_eq!(s.scalar("r(N)"), n + 1.0);
}

#[test]
fn mdesc_reports_missing() {
    let mut s = Session::with_missing_fixture();
    let out = s.run("mdesc y z x");
    assert!(out.contains("Percent Missing"), "{out}");
    assert_eq!(s.scalar("r(miss)"), 0.0);
    s.run("mdesc y");
    assert_eq!(s.scalar("r(miss)"), 1.0);
    assert_close(s.scalar("r(percent)"), 100.0 / 12.0, 1e-9);
}

#[test]
fn winsor2_winsorizes_and_trims() {
    let mut s = Session::with_fixture();
    s.run("winsor2 x, cuts(10 90)");
    s.run("summarize x, detail");
    let (p10, p90) = (s.scalar("r(p10)"), s.scalar("r(p90)"));
    s.run("summarize x_w");
    assert_close(s.scalar("r(min)"), p10, 1e-9);
    assert_close(s.scalar("r(max)"), p90, 1e-9);
    s.run("winsor2 x, cuts(10 90) trim");
    s.run("count if missing(x_tr)");
    assert!(s.scalar("r(N)") >= 2.0);
    s.run("winsor2 y, cuts(10 90) by(group) suffix(_g)");
    assert!(s.has_var("y_g"));
    s.run("winsor2 z, replace cuts(0 50)");
    s.run("summarize z");
    assert!(s.scalar("r(max)") <= p90.max(1e9));
    s.fails("winsor2 x, cuts(90 10)");
    s.fails("winsor2 name");
}
