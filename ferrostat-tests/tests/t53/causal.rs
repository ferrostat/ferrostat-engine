//! Endogenous-treatment regressions, control-function effects,
//! heterogeneous DID, switching DID, gencohort and treatment effects after
//! the extended regression models.

#[allow(unused_imports)]
use ferrostat_tests::*;

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol, "{what}: {a} vs {b} (tol {tol})");
}

/// Cross section with an exogenous binary treatment (independent errors).
fn exog_data(s: &mut Session, n: usize) {
    s.script(&format!(
        "clear
         set obs {n}
         set seed 2024
         gen x = rnormal()
         gen z = rnormal()
         gen d = (0.2 + z + .3*x + rnormal()) > 0
         gen y = 1 + .5*x + 2*d + rnormal()
         gen c = rpoisson(exp(.3 + .4*x + .5*d + .5*rnormal()))"
    ));
}

#[test]
fn etregress_without_endogeneity_matches_regress() {
    let mut s = Session::new();
    exog_data(&mut s, 2000);
    s.run("regress y x d");
    let (bx, bd) = (s.scalar("_b[x]"), s.scalar("_b[d]"));
    let out = s.run("etregress y x, treat(d = z x)");
    assert!(out.contains("rho") && out.contains("lambda"), "{out}");
    close(s.scalar("_b[y:x]"), bx, 0.05, "x");
    close(s.scalar("_b[y:1.d]"), bd, 0.1, "treatment");
    assert!(s.scalar("e(rho)").abs() < 0.15);
    let ml = s.scalar("_b[y:1.d]");
    s.run("etregress y x, treat(d = z x) twostep");
    close(s.scalar("_b[y:1.d]"), ml, 0.1, "twostep vs ml");
    s.run("predict p, ptrt");
    s.run("predict y1, yctrt");
    s.run("predict y0, ycntrt");
    s.run("summarize p");
    assert!(s.scalar("r(min)") > 0.0 && s.scalar("r(max)") < 1.0);
    s.run("etregress y x, treat(d = z x) poutcomes");
    assert!(s.scalar("e(sigma0)") > 0.5 && s.scalar("e(sigma1)") > 0.5);
}

#[test]
fn etregress_ml_recovers_endogenous_treatment_effect() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 3000
         set seed 77
         gen x = rnormal()
         gen z = rnormal()
         gen u = rnormal()
         gen e = .6*u + .8*rnormal()
         gen d = (z + .3*x + u) > 0
         gen y = 1 + .5*x + 2*d + e",
    );
    s.run("regress y x d");
    let ols = s.scalar("_b[d]");
    s.run("etregress y x, treat(d = z x)");
    let ml = s.scalar("_b[y:1.d]");
    assert!((ml - 2.0).abs() < 0.15, "{ml}");
    assert!((ols - 2.0).abs() > (ml - 2.0).abs());
    close(s.scalar("e(rho)"), 0.6, 0.12, "rho");
}

#[test]
fn etpoisson_without_endogeneity_matches_poisson_slopes() {
    let mut s = Session::new();
    exog_data(&mut s, 1500);
    s.run("poisson c x d");
    let (bx, bd) = (s.scalar("_b[x]"), s.scalar("_b[d]"));
    s.run("etpoisson c x, treat(d = z x)");
    close(s.scalar("_b[c:x]"), bx, 0.08, "x");
    close(s.scalar("_b[c:1.d]"), bd, 0.15, "d");
    s.run("predict mu");
    s.run("predict ce, cte");
}

#[test]
fn eteffects_with_exogenous_treatment_matches_regression_adjustment() {
    let mut s = Session::new();
    exog_data(&mut s, 3000);
    s.run("teffects ra (y x) (d)");
    let ra = s.scalar("_b[ATE:r1vs0.d]");
    let out = s.run("eteffects (y x) (d z x)");
    assert!(out.contains("ATE"), "{out}");
    close(s.scalar("_b[ATE:r1vs0.d]"), ra, 0.15, "ATE");
    assert!(s.scalar("_se[ATE:r1vs0.d]") > 0.0);
    s.run("eteffects (y x) (d z x), atet aequations");
    close(s.scalar("_b[ATET:r1vs0.d]"), 2.0, 0.25, "ATET");
    s.run("eteffects (y x) (d z x), pomeans");
}

/// Two periods, two groups: cohort 2 treated in period 2.
fn did_2x2(s: &mut Session) {
    s.script(
        "clear
         set obs 400
         set seed 5
         gen id = _n
         gen grp = ceil(_n/20)
         gen treated = grp <= 8
         expand 2
         bysort id: gen t = _n
         gen d = treated & t == 2
         gen y = grp/10 + .7*t + 1.5*d + rnormal()",
    );
}

fn manual_did(s: &mut Session) -> f64 {
    let mut m = [[0.0; 2]; 2];
    for g in 0..2 {
        for t in 1..=2 {
            s.run(&format!("summarize y if treated == {g} & t == {t}"));
            m[g][t - 1] = s.scalar("r(mean)");
        }
    }
    (m[1][1] - m[1][0]) - (m[0][1] - m[0][0])
}

#[test]
fn hdidregress_two_period_equals_2x2_did() {
    let mut s = Session::new();
    did_2x2(&mut s);
    let did = manual_did(&mut s);
    s.run("hdidregress ra (y) (d), group(grp) time(t)");
    s.run("estat aggregation");
    s.run("matrix B = r(b)");
    close(s.scalar("B[1,1]"), did, 1e-6, "ra");
    s.run("hdidregress twfe (y) (d), group(grp) time(t)");
    close(s.scalar("_b[2.cohort#2.t]"), did, 1e-6, "twfe");
    s.run("hdidregress aipw (y) (d), group(grp) time(t)");
    s.run("estat aggregation, cohort");
    // panel version: units are the ids
    s.run("xtset id t");
    s.run("xthdidregress twfe (y) (d), group(grp)");
    close(s.scalar("_b[2.cohort#2.t]"), did, 1e-6, "xt twfe");
    s.run("xthdidregress ra (y) (d), group(grp)");
    s.run("estat aggregation");
    s.run("matrix B = r(b)");
    close(s.scalar("B[1,1]"), did, 1e-6, "xt ra");
    s.run("xtdidregress (y) (d), group(id) time(t)");
    close(s.scalar("_b[r1vs0.d]"), did, 1e-6, "xtdidregress");
}

#[test]
fn staggered_twfe_cells_and_aggregations() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 60
         set seed 9
         gen id = _n
         gen first = cond(_n <= 20, 3, cond(_n <= 40, 4, 0))
         expand 5
         bysort id: gen t = _n
         gen d = first > 0 & t >= first
         gen y = id/30 + .2*t + 2*d + rnormal()*.1
         xtset id t",
    );
    s.run("gencohort coh = d");
    let c = s.nums("coh");
    let f = s.nums("first");
    assert_eq!(c, f);
    s.run("xthdidregress twfe (y) (d), group(id)");
    s.run("estat aggregation");
    s.run("matrix B = r(b)");
    close(s.scalar("B[1,1]"), 2.0, 0.1, "overall ATET");
    s.run("estat aggregation, dynamic");
    s.run("estat aggregation, time");
    s.run("xthdidregress aipw (y) (d), group(id)");
    s.run("estat aggregation");
    s.run("matrix B = r(b)");
    close(s.scalar("B[1,1]"), 2.0, 0.1, "aipw overall");
    s.run("estat ptrends");
}

#[test]
fn xtswitchdid_equals_did_without_switching_off() {
    let mut s = Session::new();
    did_2x2(&mut s);
    let did = manual_did(&mut s);
    s.run("xtset id t");
    s.run("xtswitchdid (y) (d), reps(20) rseed(1)");
    close(s.scalar("_b[ATET:DID_M]"), did, 1e-8, "DID_M");
    // switching off: leavers mirror joiners
    s.script(
        "clear
         set obs 200
         set seed 3
         gen id = _n
         expand 3
         bysort id: gen t = _n
         gen d = (id <= 50 & t >= 2) | (id > 150 & t <= 2)
         gen y = id/100 + .5*t + 1*d + rnormal()*.05
         xtset id t",
    );
    s.run("xtswitchdid (y) (d), reps(10) rseed(2)");
    close(s.scalar("_b[ATET:DID_M]"), 1.0, 0.05, "with leavers");
    assert!(s.scalar("e(N_switch_off)") > 0.0);
}

#[test]
fn estat_teffects_after_erm() {
    let mut s = Session::new();
    exog_data(&mut s, 800);
    s.run("eregress y x, extreat(d)");
    s.run("estat teffects");
    s.run("matrix B = r(b)");
    let ate = s.scalar("B[1,1]");
    s.run("summarize x");
    let mx = s.scalar("r(mean)");
    let manual = (s.scalar("_b[y:1.d]") - s.scalar("_b[y:0.d]")) + (s.scalar("_b[y:1.d#c.x]") - s.scalar("_b[y:0.d#c.x]")) * mx;
    close(ate, manual, 1e-6, "ATE");
    s.run("predict te, te");
    s.run("summarize te");
    close(s.scalar("r(mean)"), manual, 1e-6, "mean te");
    s.run("estat teffects, pomean");
    s.run("estat teffects, atet");
    // endogenous treatment, probit outcome
    s.run("gen yb = y > 2");
    s.run("eprobit yb x, entreat(d = z x)");
    s.run("estat teffects");
    s.run("matrix B = r(b)");
    let a = s.scalar("B[1,1]");
    assert!(a > 0.0 && a < 1.0, "{a}");
    s.run("estat teffects, atet");
    s.run("predict p0, pr base(d = 0)");
    s.run("predict p1, pr base(d = 1)");
    s.run("gen dp = p1 - p0");
    s.run("predict te2, te");
    s.run("summarize dp");
    let m1 = s.scalar("r(mean)");
    s.run("summarize te2");
    close(s.scalar("r(mean)"), m1, 1e-6, "te vs base()");
    s.run("predict pm, pomean tlevel(0)");
    // eoprobit category probabilities sum to one
    s.run("gen yo = 1 + (y > 1) + (y > 3)");
    s.run("eoprobit yo x, extreat(d)");
    s.run("predict q1 q2 q3, pr");
    s.run("gen qs = q1 + q2 + q3");
    s.run("summarize qs");
    close(s.scalar("r(min)"), 1.0, 1e-8, "sum");
    close(s.scalar("r(max)"), 1.0, 1e-8, "sum");
    s.run("estat teffects, outcome(2)");
}
