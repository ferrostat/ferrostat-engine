//! Heterogeneity-robust difference-in-differences: `drdid`, `csdid` (with
//! `estat` aggregations and `csdid_plot`), `did_imputation` (with
//! `event_plot`), `eventstudyinteract`, and `didregress` / `xtdidregress`.
//!
//! Reference values were produced by independent implementations of the
//! same estimators (a separate Python implementation of the 2x2 doubly
//! robust estimators and of the group-time aggregations, and a dense-matrix
//! implementation of the imputation estimator) on the deterministic designs
//! below, and by identities that must hold exactly (no staggering, single
//! cohorts, hand-computed imputation).

use ferrostat_tests::*;

/// 400 units observed in two periods; treatment depends on x.
fn two_by_two(s: &mut Session) {
    s.script(
        "clear
         set obs 400
         gen id = _n
         gen double x = mod(_n*37,101)/101
         gen d = mod(_n*13,7) < 3 | (x > 0.8 & mod(_n,5)==0)
         expand 2
         sort id
         by id: gen t = _n
         gen double y = 1 + 2*x + 0.5*t + 0.3*d + d*(t==2)*(1 + 0.7*x) + 0.4*x*(t==2) + mod(_n*7919,1009)/1009 - 0.5",
    );
}

/// 120 units, 6 periods, cohorts treated in periods 3, 4, 5 and never;
/// effects grow with time since treatment.
fn staggered(s: &mut Session) {
    s.script(
        "clear
         set obs 120
         gen id = _n
         gen g = cond(mod(_n,4)==0, 0, cond(mod(_n,4)==1, 3, cond(mod(_n,4)==2, 4, 5)))
         gen double a = mod(_n*37,101)/101
         gen double x = mod(_n*53,97)/97
         expand 6
         sort id
         by id: gen t = _n
         gen double y = a + 0.2*t + 0.5*x*t/6 + (g>0 & t>=g)*(1 + 0.3*(t-g)) + mod(_n*7919,1009)/1009 - 0.5
         gen Ei = g if g>0",
    );
}

#[test]
fn drdid_panel_estimators_match_reference() {
    let mut s = Session::new();
    two_by_two(&mut s);
    let out = s.run("drdid y x, ivar(id) time(t) tr(d) dripw");
    assert!(out.contains("Doubly robust difference-in-differences"), "{out}");
    assert!(out.contains("Outcome model  : least squares"), "{out}");
    assert!(out.contains("Treatment model: inverse probability"), "{out}");
    assert!(out.contains("(1 vs 0)"), "{out}");
    assert_close(s.scalar("_b[ATET:r1vs0.d]"), 1.36630673, 1e-7);
    assert_close(s.scalar("_se[ATET:r1vs0.d]"), 0.03877829, 1e-7);
    assert_eq!(s.scalar("e(N)"), 800.0);
    assert_eq!(s.run("display e(method)").trim(), "dripw");
    assert_eq!(s.run("display e(datatype)").trim(), "panel");
    s.run("qui drdid y x, ivar(id) time(t) tr(d) reg");
    assert_close(s.scalar("e(att1)"), 1.36620990, 1e-7);
    assert_close(s.scalar("sqrt(e(attvar1))"), 0.03874975, 1e-7);
    s.run("qui drdid y x, ivar(id) time(t) tr(d) stdipw");
    assert_close(s.scalar("e(att1)"), 1.36723642, 1e-7);
    assert_close(s.scalar("sqrt(e(attvar1))"), 0.03895174, 1e-7);
    // default estimator is the improved doubly robust one
    let out = s.run("drdid y x, ivar(id) time(t) tr(d)");
    assert!(out.contains("No estimator selected. Using default drimp"), "{out}");
    assert!(out.contains("inverse probability tilting"), "{out}");
    assert_eq!(s.run("display e(method)").trim(), "drimp");
}

#[test]
fn drdid_repeated_cross_sections_match_reference() {
    let mut s = Session::new();
    two_by_two(&mut s);
    s.run("qui drdid y x, time(t) tr(d) dripw");
    assert_close(s.scalar("e(att1)"), 1.36630673, 1e-7);
    assert_close(s.scalar("sqrt(e(attvar1))"), 0.04265312, 1e-7);
    assert_eq!(s.run("display e(datatype)").trim(), "rcs");
    s.run("qui drdid y x, time(t) tr(d) reg");
    assert_close(s.scalar("sqrt(e(attvar1))"), 0.09079418, 1e-7);
    s.run("qui drdid y x, time(t) tr(d) stdipw");
    assert_close(s.scalar("sqrt(e(attvar1))"), 0.10785810, 1e-7);
    // all estimators side by side
    let out = s.run("drdid y x, time(t) tr(d) all");
    assert!(out.contains("dripw_rc1"), "{out}");
    assert!(out.contains("Doubly robust difference-in-differences estimator summary"), "{out}");
}

#[test]
fn drdid_without_covariates_is_the_difference_in_means() {
    let mut s = Session::new();
    two_by_two(&mut s);
    s.script(
        "sort id t
         by id: gen double dy = y - y[_n-1] if t==2
         qui summarize dy if d==1
         scalar m1 = r(mean)
         qui summarize dy if d==0
         scalar m0 = r(mean)",
    );
    let did = s.scalar("m1 - m0");
    for m in ["reg", "dripw", "drimp", "stdipw", "ipw"] {
        s.run(&format!("qui drdid y, ivar(id) time(t) tr(d) {m}"));
        assert_close(s.scalar("e(att1)"), did, 1e-9);
    }
    // stub(): the recentred influence function averages to the ATT
    s.run("qui drdid y, ivar(id) time(t) tr(d) reg stub(__)");
    s.run("qui summarize __att");
    assert_close(s.scalar("r(mean)"), did, 1e-9);
    assert_eq!(s.scalar("r(N)"), 400.0);
}

#[test]
fn drdid_wild_bootstrap_is_reproducible() {
    let mut s = Session::new();
    two_by_two(&mut s);
    s.run("qui drdid y x, ivar(id) time(t) tr(d) dripw wboot(reps(199)) rseed(42)");
    let se1 = s.scalar("_se[ATET:r1vs0.d]");
    s.run("qui drdid y x, ivar(id) time(t) tr(d) dripw wboot(reps(199)) rseed(42)");
    assert_eq!(s.scalar("_se[ATET:r1vs0.d]"), se1);
    // bootstrap and analytic standard errors agree roughly
    assert!((se1 / 0.03877829 - 1.0).abs() < 0.2, "{se1}");
    assert_eq!(s.run("display e(vcetype)").trim(), "WBoot");
}

#[test]
fn csdid_matches_reference_aggregations() {
    let mut s = Session::new();
    staggered(&mut s);
    let out = s.run("csdid y, ivar(id) time(t) gvar(g)");
    assert!(out.contains("Difference-in-difference with Multiple Time Periods"), "{out}");
    assert!(out.contains("g3"), "{out}");
    assert!(out.contains("t_2_3"), "{out}");
    assert!(out.contains("Control: Never Treated"), "{out}");
    assert_eq!(s.run("display e(cmd)").trim(), "csdid");
    let out = s.run("estat simple");
    s.run("matrix RB = r(b)");
    s.run("matrix RV = r(V)");
    assert!(out.contains("Average Treatment Effect on Treated"), "{out}");
    assert_close(s.scalar("el(RB,1,1)"), 1.34425, 5e-6);
    assert_close(s.scalar("sqrt(el(RV,1,1))"), 0.070736, 5e-6);
    s.run("estat event");
    s.run("matrix RB = r(b)");
    s.run("matrix RV = r(V)");
    let b = s.run("matrix list RB");
    assert!(b.contains("Tm3") && b.contains("Tp3"), "{b}");
    assert_close(s.scalar("el(RB,1,2)"), 1.461465, 5e-6); // Post_avg
    assert_close(s.scalar("sqrt(el(RV,2,2))"), 0.0789679, 5e-6);
    assert_close(s.scalar("el(RB,1,6)"), 1.012638, 5e-6); // Tp0
    assert_close(s.scalar("sqrt(el(RV,6,6))"), 0.0510409, 5e-6);
    s.run("estat group");
    s.run("matrix RB = r(b)");
    s.run("matrix RV = r(V)");
    assert_close(s.scalar("el(RB,1,2)"), 1.455727, 5e-6); // G3
    assert_close(s.scalar("sqrt(el(RV,2,2))"), 0.102747, 5e-6);
    let out = s.run("estat pretrend");
    assert!(out.contains("Pretrend Test"), "{out}");
    // covariates (doubly robust), panel and repeated cross sections
    s.run("qui csdid y x, ivar(id) time(t) gvar(g) method(dripw) agg(simple)");
    assert_close(s.scalar("_b[ATT]"), 1.33589, 5e-6);
    assert_close(s.scalar("_se[ATT]"), 0.0651617, 5e-6);
    // not-yet-treated comparison group
    s.run("qui csdid y, ivar(id) time(t) gvar(g) notyet agg(simple)");
    assert_close(s.scalar("_b[ATT]"), 1.343267, 5e-6);
    assert_close(s.scalar("_se[ATT]"), 0.0687966, 5e-6);
    assert_eq!(s.run("display e(control_group)").trim(), "Not yet Treated");
    // agg() posts the aggregation
    s.run("qui csdid y, ivar(id) time(t) gvar(g) agg(event)");
    assert_close(s.scalar("_b[Tp1]"), 1.303055, 5e-6);
    assert_close(s.scalar("_se[Tp1]"), 0.0727463, 5e-6);
}

#[test]
fn csdid_without_staggering_equals_two_by_two_did() {
    let mut s = Session::new();
    staggered(&mut s);
    s.script(
        "keep if g==0 | g==4
         sort id t
         by id: gen double d4 = y - y[_n-1] if t==4
         qui summarize d4 if g==4
         scalar a1 = r(mean)
         qui summarize d4 if g==0
         scalar a0 = r(mean)",
    );
    s.run("qui csdid y, ivar(id) time(t) gvar(g)");
    assert_close(s.scalar("_b[g4:t_3_4]"), s.scalar("a1 - a0"), 1e-10);
    // equals drdid on the same 2x2 sample
    s.run("gen tr = g==4");
    s.run("qui drdid y if t==3 | t==4, ivar(id) time(t) tr(tr) reg");
    let att = s.scalar("e(att1)");
    let se = s.scalar("sqrt(e(attvar1))");
    s.run("qui csdid y, ivar(id) time(t) gvar(g)");
    assert_close(s.scalar("_b[g4:t_3_4]"), att, 1e-10);
    assert_close(s.scalar("_se[g4:t_3_4]"), se, 1e-10);
    // and with a single cohort the event-time effect equals the cell
    s.run("estat event");
    s.run("matrix RB = r(b)");
    // Pre_avg Post_avg Tm2 Tm1 Tp0 ...
    assert_close(s.scalar("el(RB,1,5)"), att, 1e-10);
}

#[test]
fn csdid_plot_and_wild_bootstrap() {
    let mut s = Session::new();
    staggered(&mut s);
    s.run("qui csdid y, ivar(id) time(t) gvar(g) agg(event) wboot(reps(99)) rseed(7)");
    let se = s.scalar("_se[Tp0]");
    s.run("qui csdid y, ivar(id) time(t) gvar(g) agg(event) wboot(reps(99)) rseed(7)");
    assert_eq!(s.scalar("_se[Tp0]"), se);
    let cb = s.run("matrix list e(cband)");
    assert!(cb.contains("ll") && cb.contains("ul"), "{cb}");
    let dir = temp_dir("t40_csdid_plot");
    let f = dir.join("cs.svg");
    let out = s.run(&format!("csdid_plot, saving(\"{}\")", f.display()));
    assert!(out.contains("saved"), "{out}");
}

#[test]
fn did_imputation_matches_reference_and_hand_computation() {
    let mut s = Session::new();
    staggered(&mut s);
    let out = s.run("did_imputation y id t Ei, minn(0)");
    assert!(out.contains("tau"), "{out}");
    let tau = s.scalar("_b[tau]");
    assert_close(tau, 1.33729795, 1e-6);
    assert_close(s.scalar("_se[tau]"), 0.06506451, 1e-6);
    assert_eq!(s.run("display e(cmd)").trim(), "did_imputation");
    // hand computation: fit unit and period effects on untreated cells
    s.script(
        "gen D = !missing(Ei) & t >= Ei
         qui regress y i.id i.t if D==0
         predict double y0
         gen double te = y - y0 if D==1
         qui summarize te",
    );
    assert_close(tau, s.scalar("r(mean)"), 1e-6);
    s.run("qui did_imputation y id t Ei, horizons(0/2) minn(0)");
    assert_close(s.scalar("_b[tau0]"), 1.01034269, 1e-6);
    assert_close(s.scalar("_se[tau0]"), 0.05271543, 1e-6);
    assert_close(s.scalar("_b[tau2]"), 1.60876527, 1e-6);
    assert_close(s.scalar("_se[tau2]"), 0.09346117, 1e-6);
    let tau1 = s.scalar("_b[tau1]");
    s.run("qui summarize te if t - Ei == 1");
    assert_close(tau1, s.scalar("r(mean)"), 1e-6);
    // pre-trend coefficients come from the untreated-only regression
    s.run("qui did_imputation y id t Ei, minn(0) pretrends(2)");
    let p1 = s.scalar("_b[pre1]");
    assert!(s.scalar("e(pre_p)") > 0.05);
    s.script(
        "gen byte lead1 = t - Ei == -1
         gen byte lead2 = t - Ei == -2
         qui regress y lead1 lead2 i.id i.t if D==0",
    );
    assert_close(p1, s.scalar("_b[lead1]"), 1e-6);
    // event_plot draws tau# and pre# coefficients
    s.run("qui did_imputation y id t Ei, horizons(0/2) pretrends(2) minn(0)");
    let dir = temp_dir("t40_event_plot");
    let out = s.run(&format!("event_plot, saving(\"{}\")", dir.join("ep.svg").display()));
    assert!(out.contains("saved"), "{out}");
}

#[test]
fn did_imputation_autosample_and_minn() {
    let mut s = Session::new();
    staggered(&mut s);
    // units treated in every period cannot be imputed
    s.run("replace Ei = 1 if id <= 4 & g==3");
    s.fails("did_imputation y id t Ei");
    assert!(s.has_var("cannot_impute"));
    let out = s.run("did_imputation y id t Ei, autosample minn(0)");
    assert!(out.contains("tau"), "{out}");
    // a small effective sample suppresses the coefficient by default
    let out = s.run("did_imputation y id t Ei if id > 4, horizons(0/3)");
    assert!(out.contains("insufficient effective sample size"), "{out}");
}

#[test]
fn eventstudyinteract_equals_cohort_effects() {
    let mut s = Session::new();
    staggered(&mut s);
    s.script(
        "gen rel = t - g if g>0
         gen byte F3 = rel==-3
         gen byte F2 = rel==-2
         gen byte L0 = rel==0
         gen byte L1 = rel==1
         gen byte L2 = rel==2
         gen byte L3 = rel==3
         gen byte F4 = rel==-4
         gen byte never = g==0",
    );
    let out = s.run("eventstudyinteract y F4 F3 F2 L0 L1 L2 L3, cohort(g) control_cohort(never) absorb(id t) vce(cluster id)");
    assert!(out.contains("IW estimates for dynamic effects"), "{out}");
    s.run("matrix BIW = e(b_iw)");
    let b_iw0 = s.scalar("el(BIW,1,4)");
    // with never-treated controls and all relative periods but -1, the
    // post-treatment IW estimates equal the group-time aggregation
    s.run("qui csdid y, ivar(id) time(t) gvar(g) agg(event)");
    assert_close(b_iw0, s.scalar("_b[Tp0]"), 1e-8);
    // single cohort: the IW estimate is the interacted coefficient
    s.run("qui eventstudyinteract y L0 L1 L2 if g==0 | g==4, cohort(g) control_cohort(never) absorb(id t) vce(cluster id)");
    s.run("matrix BIW = e(b_iw)");
    let b = s.scalar("el(BIW,1,2)");
    s.run("qui reghdfe y L0 L1 L2 if g==0 | g==4, absorb(id t) vce(cluster id)");
    assert_close(b, s.scalar("_b[L1]"), 1e-8);
}

#[test]
fn didregress_classic_and_panel() {
    let mut s = Session::new();
    staggered(&mut s);
    s.run("gen byte treated = g==4 & t>=4");
    let out = s.run("didregress (y) (treated) if g==0 | g==4, group(id) time(t)");
    assert!(out.contains("Number of groups and treatment time"), "{out}");
    assert!(out.contains("Difference-in-differences regression"), "{out}");
    assert!(out.contains("ATET"), "{out}");
    let atet = s.scalar("_b[ATET:r1vs0.treated]");
    s.run("qui reghdfe y treated if g==0 | g==4, absorb(id t) vce(cluster id)");
    assert_close(atet, s.scalar("_b[treated]"), 1e-9);
    s.run("qui didregress (y) (treated) if g==0 | g==4, group(id) time(t)");
    let out = s.run("estat ptrends");
    assert!(out.contains("H0: Linear trends are parallel"), "{out}");
    let out = s.run("estat granger");
    assert!(out.contains("H0: No effect in anticipation of treatment"), "{out}");
    s.run("xtset id t");
    let out = s.run("xtdidregress (y x) (treated) if g==0 | g==4, group(id) time(t)");
    assert!(out.contains("Longitudinal"), "{out}");
    let out = s.run("didregress (y) (treated) if g==0 | g==4, group(id) time(t) wildbootstrap(rseed(3) reps(99))");
    assert!(out.contains("Wild-cluster bootstrap"), "{out}");
}

#[test]
fn did_packages_are_built_in() {
    let mut s = Session::new();
    for p in ["csdid", "drdid", "did_imputation", "eventstudyinteract"] {
        let out = s.run(&format!("ssc install {p}"));
        assert!(!out.to_lowercase().contains("not found"), "{p}: {out}");
    }
}
