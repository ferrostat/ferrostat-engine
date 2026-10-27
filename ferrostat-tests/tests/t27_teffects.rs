//! `teffects ra|ipw|aipw`: point estimates equal the textbook formulas
//! computed step by step with other commands.

use ferrostat_tests::*;

fn data() -> Session {
    let mut s = Session::new();
    s.script("set obs 400\nset seed 12\ngen x = rnormal()\ngen z = runiform()\ngen d = runiform() < invlogit(0.5*x - 0.3)\ngen y = 1 + 2*d + x + 0.5*z + d*x + rnormal()");
    s
}

#[test]
fn ra_matches_separate_regressions() {
    let mut s = data();
    s.run("teffects ra (y x z) (d)");
    let ate = s.scalar("_b[ATE:r1vs0.d]");
    let pom0 = s.scalar("_b[POmean:0.d]");
    s.script("quietly regress y x z if d==0\npredict m0\nquietly regress y x z if d==1\npredict m1\ngen te = m1 - m0");
    s.run("summarize te, meanonly");
    assert_close(ate, s.scalar("r(mean)"), 1e-8);
    s.run("summarize m0, meanonly");
    assert_close(pom0, s.scalar("r(mean)"), 1e-8);
    s.run("teffects ra (y x z) (d), atet");
    s.run("summarize te if d==1, meanonly");
    assert_close(s.scalar("_b[ATET:r1vs0.d]"), s.scalar("r(mean)"), 1e-8);
    assert!(s.scalar("_se[ATET:r1vs0.d]") > 0.0);
}

#[test]
fn ipw_matches_weighted_means() {
    let mut s = data();
    let out = s.run("teffects ipw (y) (d x z)");
    assert!(out.contains("inverse-probability weights") && out.contains("Treatment model: logit"), "{out}");
    let ate = s.scalar("_b[ATE:r1vs0.d]");
    s.script("quietly logit d x z\npredict p, pr\ngen w = cond(d, 1/p, 1/(1-p))");
    s.run("summarize y [aw=w] if d==1, meanonly");
    let m1 = s.scalar("r(mean)");
    s.run("summarize y [aw=w] if d==0, meanonly");
    let m0 = s.scalar("r(mean)");
    assert_close(ate, m1 - m0, 1e-6);
}

#[test]
fn aipw_and_pomeans_consistent() {
    let mut s = data();
    s.run("teffects aipw (y x z) (d x z)");
    let ate = s.scalar("_b[ATE:r1vs0.d]");
    s.run("teffects aipw (y x z) (d x z), pomeans");
    assert_close(s.scalar("_b[POmeans:1.d]") - s.scalar("_b[POmeans:0.d]"), ate, 1e-8);
    s.fails("teffects aipw (y x z) (d x z), atet");
    s.fails("teffects ra (y x z) (x)");
}
