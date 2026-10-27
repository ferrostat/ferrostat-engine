//! `reghdfe`: linear regression absorbing several sets of fixed effects.
//!
//! Reference values come from the equivalent dummy-variable regressions
//! (`regress y x i.fe1 i.fe2`, `areg`), whose coefficients and standard
//! errors must coincide once the absorbed degrees of freedom are counted the
//! same way. Multiway clustering is checked against the Cameron-Gelbach-Miller
//! inclusion-exclusion formula assembled from one-way clustered variances.

use ferrostat_tests::*;

/// Unbalanced panel: 40 ids, 8 years, two crossed cluster variables.
fn panel(s: &mut Session) {
    s.script(
        "clear
         set obs 500
         set seed 2024
         gen id = 1 + floor(runiform()*40)
         gen yr = 1 + floor(runiform()*8)
         gen c1 = 1 + floor(runiform()*12)
         gen c2 = 1 + floor(runiform()*9)
         gen z = runiform()
         gen x1 = rnormal() + c1/5
         gen x2 = rnormal() + id/10
         gen y = 1 + 0.5*x1 - 0.3*x2 + id/7 + yr/3 + rnormal() + c1/4 + (c2/5)*rnormal()",
    );
}

fn coefs_and_ses(s: &mut Session, vars: &[&str]) -> (Vec<f64>, Vec<f64>) {
    let b = vars.iter().map(|v| s.scalar(&format!("_b[{v}]"))).collect();
    let se = vars.iter().map(|v| s.scalar(&format!("_se[{v}]"))).collect();
    (b, se)
}

#[test]
fn two_way_matches_dummy_variable_regression() {
    let mut s = Session::new();
    panel(&mut s);
    s.run("regress y x1 x2 i.id i.yr");
    let (b0, se0) = coefs_and_ses(&mut s, &["x1", "x2"]);
    let (df0, r20, r2a0, rmse0) = (s.scalar("e(df_r)"), s.scalar("e(r2)"), s.scalar("e(r2_a)"), s.scalar("e(rmse)"));
    let out = s.run("reghdfe y x1 x2, absorb(id yr)");
    let (b, se) = coefs_and_ses(&mut s, &["x1", "x2"]);
    assert_all_close(&b, &b0, 1e-7);
    assert_all_close(&se, &se0, 1e-7);
    assert_close(s.scalar("e(df_r)"), df0, 1e-12);
    assert_close(s.scalar("e(r2)"), r20, 1e-8);
    assert_close(s.scalar("e(r2_a)"), r2a0, 1e-8);
    assert_close(s.scalar("e(rmse)"), rmse0, 1e-8);
    assert_eq!(s.scalar("e(N_hdfe)"), 2.0);
    assert_eq!(s.scalar("e(df_a)"), 40.0 + 8.0 - 1.0);
    assert!(out.contains("HDFE Linear regression"), "{out}");
    assert!(out.contains("Absorbing 2 HDFE groups"), "{out}");
    assert!(out.contains("Within R-sq."), "{out}");
    assert!(out.contains("Absorbed degrees of freedom:"), "{out}");
    assert_eq!(row(&out, "yr").unwrap(), vec![8.0, 1.0, 7.0]);
    assert_eq!(row(&out, "id").unwrap(), vec![40.0, 0.0, 40.0]);
    // within R-squared: 1 - RSS / (sum of squares of y net of the fixed effects)
    s.run("quietly regress y i.id i.yr");
    let rss_fe = s.scalar("e(rss)");
    s.run("quietly reghdfe y x1 x2, absorb(id yr)");
    assert_close(s.scalar("e(r2_within)"), 1.0 - s.scalar("e(rss)") / rss_fe, 1e-8);
    assert_eq!(s.run("display e(cmd)").trim(), "reghdfe");
    assert_eq!(s.run("display e(absvars)").trim(), "id yr");
}

#[test]
fn one_way_matches_areg() {
    let mut s = Session::new();
    panel(&mut s);
    s.run("areg y x1 x2, absorb(id)");
    let (b0, se0) = coefs_and_ses(&mut s, &["x1", "x2", "_cons"]);
    let out = s.run("reghdfe y x1 x2, absorb(id)");
    let (b, se) = coefs_and_ses(&mut s, &["x1", "x2", "_cons"]);
    assert_all_close(&b, &b0, 1e-9);
    assert_all_close(&se, &se0, 1e-9);
    assert!(out.contains("(MWFE estimator converged in 1 iterations)"), "{out}");
    assert!(out.contains("Absorbing 1 HDFE group "), "{out}");
}

#[test]
fn robust_and_cluster_match_dummy_regression() {
    let mut s = Session::new();
    panel(&mut s);
    s.run("regress y x1 x2 i.id i.yr, vce(robust)");
    let (_, se0) = coefs_and_ses(&mut s, &["x1", "x2"]);
    let out = s.run("reghdfe y x1 x2, absorb(id yr) vce(robust)");
    let (_, se) = coefs_and_ses(&mut s, &["x1", "x2"]);
    assert_all_close(&se, &se0, 1e-7);
    assert!(out.contains("Statistics robust to heteroskedasticity"), "{out}");
    assert!(out.contains("Robust"), "{out}");
    // c1 does not nest any fixed effect: same small-sample factor as regress
    s.run("regress y x1 x2 i.id i.yr, vce(cluster c1)");
    let (_, se0) = coefs_and_ses(&mut s, &["x1", "x2"]);
    let out = s.run("reghdfe y x1 x2, absorb(id yr) vce(cluster c1)");
    let (_, se) = coefs_and_ses(&mut s, &["x1", "x2"]);
    assert_all_close(&se, &se0, 1e-7);
    assert_eq!(s.scalar("e(N_clust)"), 12.0);
    assert_eq!(s.scalar("e(df_r)"), 11.0);
    assert!(out.contains("Number of clusters (c1)"), "{out}");
    assert!(out.contains("(Std. err. adjusted for 12 clusters in c1)"), "{out}");
}

#[test]
fn fixed_effect_nested_in_cluster_uses_no_degrees_of_freedom() {
    let mut s = Session::new();
    panel(&mut s);
    // with id nested in the cluster variable its 40 levels are redundant:
    // K = 2 regressors + intercept, as in a within regression
    let out = s.run("reghdfe y x1 x2, absorb(id) vce(cluster id)");
    assert_eq!(row(&out, "id").unwrap(), vec![40.0, 40.0, 0.0]);
    assert!(out.contains("* = FE nested within cluster"), "{out}");
    assert_eq!(s.scalar("e(df_a)"), 0.0);
    let se_hdfe = s.scalar("_se[x1]");
    s.run("regress y x1 x2 i.id, vce(cluster id)");
    let se_dummy = s.scalar("_se[x1]");
    let n = 500.0;
    let k_dummy = s.scalar("e(rank)");
    // only the (N-1)/(N-K) factor differs
    let ratio = ((n - 1.0) / (n - 3.0) / ((n - 1.0) / (n - k_dummy))).sqrt();
    assert_close(se_hdfe, se_dummy * ratio, 1e-7);
}

#[test]
fn multiway_clustering_is_cameron_gelbach_miller() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 600
         set seed 7
         gen id = 1 + floor(runiform()*30)
         gen yr = 1 + floor(runiform()*8)
         gen c1 = 1 + floor(runiform()*12)
         gen c2 = 1 + floor(runiform()*9)
         gen x1 = rnormal() + c1/5
         gen x2 = rnormal() + id/10
         gen y = 1 + 0.5*x1 - 0.3*x2 + id/7 + yr/3 + rnormal() + c1/4 + c2/5*rnormal()
         gen c12 = c1*100 + c2",
    );
    let n = 600.0;
    // one-way clustered variances of the dummy regression, with their own
    // small-sample factors removed
    let mut raw = Vec::new();
    for c in ["c1", "c2", "c12"] {
        s.run(&format!("quietly regress y x1 x2 i.id i.yr, vce(cluster {c})"));
        let g = s.scalar("e(N_clust)");
        let k = s.scalar("e(rank)");
        let q = (n - 1.0) / (n - k) * g / (g - 1.0);
        s.run("matrix V = e(V)");
        raw.push([s.scalar("V[1,1]") / q, s.scalar("V[2,2]") / q, s.scalar("V[1,2]") / q, k]);
    }
    let k = raw[0][3];
    let gmin: f64 = 9.0;
    let q = (n - 1.0) / (n - k) * gmin / (gmin - 1.0);
    let v11 = q * (raw[0][0] + raw[1][0] - raw[2][0]);
    let v22 = q * (raw[0][1] + raw[1][1] - raw[2][1]);
    let v12 = q * (raw[0][2] + raw[1][2] - raw[2][2]);
    let out = s.run("reghdfe y x1 x2, absorb(id yr) vce(cluster c1 c2)");
    s.run("matrix V = e(V)");
    assert_close(s.scalar("V[1,1]"), v11, 1e-7);
    assert_close(s.scalar("V[2,2]"), v22, 1e-7);
    assert_close(s.scalar("V[1,2]"), v12, 1e-7);
    assert!(!out.contains("non-positive semi-definite"), "{out}");
    // independent numpy computation on the same data (dummy-variable
    // regression, CGM meat, (N-1)/(N-K) * G_min/(G_min-1))
    assert_close(s.scalar("_se[x1]"), 0.0873937, 1e-5);
    assert_close(s.scalar("_se[x2]"), 0.06486725, 1e-5);
    assert_eq!(s.scalar("e(N_clust)"), 9.0);
    assert_eq!(s.scalar("e(N_clust1)"), 12.0);
    assert_eq!(s.scalar("e(df_r)"), 8.0);
    assert!(out.contains("Number of clusters (c1)") && out.contains("Number of clusters (c2)"), "{out}");
}

#[test]
fn singletons_are_dropped_iteratively() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 12
         gen id = cond(_n <= 10, 1 + mod(_n, 3), _n)
         gen yr = cond(_n <= 10, 1 + mod(_n, 2), 1)
         gen x = mod(_n*7, 5) + _n/10
         gen y = x + id + mod(_n*3, 4)",
    );
    // obs 11 and 12 are alone in their id
    let out = s.run("reghdfe y x, absorb(id yr)");
    assert!(out.contains("(dropped 2 singleton observations)"), "{out}");
    assert_eq!(s.scalar("e(N)"), 10.0);
    assert_eq!(s.scalar("e(num_singletons)"), 2.0);
    s.run("count if e(sample)");
    assert_eq!(s.scalar("r(N)"), 10.0);
    let out = s.run("reghdfe y x, absorb(id yr) keepsingletons");
    assert!(!out.contains("singleton"), "{out}");
    assert_eq!(s.scalar("e(N)"), 12.0);
}

#[test]
fn redundant_levels_follow_connected_components() {
    let mut s = Session::new();
    // two disconnected blocks of (a, b) pairs: one extra redundant level
    s.script(
        "clear
         set obs 60
         gen blk = _n > 30
         gen a = 1 + mod(_n, 5) + 5*blk
         gen b = 1 + mod(_n*7, 3) + 3*blk
         gen x = sin(_n) + a/3
         gen y = 2*x + a/2 - b/4 + cos(_n*3)",
    );
    let out = s.run("reghdfe y x, absorb(a b)");
    assert_eq!(row(&out, "b").unwrap(), vec![6.0, 2.0, 4.0]);
    let se = s.scalar("_se[x]");
    let b = s.scalar("_b[x]");
    s.run("regress y x i.a i.b");
    assert_close(b, s.scalar("_b[x]"), 1e-8);
    assert_close(se, s.scalar("_se[x]"), 1e-8);
    // a third set of fixed effects is counted conservatively
    s.run("gen c = 1 + mod(_n, 4)");
    let out = s.run("reghdfe y x, absorb(a b c)");
    assert_eq!(row(&out, "c").unwrap(), vec![4.0, 1.0, 3.0]);
    assert!(out.contains("? = number of redundant parameters may be higher"), "{out}");
    let b3 = s.scalar("_b[x]");
    s.run("regress y x i.a i.b i.c");
    assert_close(b3, s.scalar("_b[x]"), 1e-7);
}

#[test]
fn predictions_and_saved_fixed_effects() {
    let mut s = Session::new();
    panel(&mut s);
    s.run("reghdfe y x1 x2, absorb(fid=id fyr=yr) residuals(e)");
    s.script(
        "predict double xb, xb
         predict double d, d
         predict double xbd, xbd
         predict double r, residuals
         gen double chk1 = abs(fid + fyr - d)
         gen double chk2 = abs(xbd - xb - d)
         gen double chk3 = abs(y - xbd - e)
         gen double chk4 = abs(r - e)",
    );
    for v in ["chk1", "chk2", "chk3", "chk4"] {
        s.run(&format!("summarize {v}"));
        assert!(s.scalar("r(max)") < 1e-7, "{v}: {}", s.scalar("r(max)"));
    }
    // fitted values equal those of the dummy-variable regression
    s.run("regress y x1 x2 i.id i.yr");
    s.run("predict double yhat, xb");
    s.run("gen double chk5 = abs(yhat - xbd)");
    s.run("summarize chk5");
    assert!(s.scalar("r(max)") < 1e-7);
    // d needs the residuals
    s.run("reghdfe y x1 x2, absorb(id yr)");
    s.fails("predict double d2, d");
}

#[test]
fn savefe_and_interactions() {
    let mut s = Session::new();
    panel(&mut s);
    s.run("reghdfe y x1 x2, absorb(id yr, savefe)");
    assert!(s.has_var("__hdfe1__") && s.has_var("__hdfe2__"));
    // i.id#i.yr is the same as one grouping of the pairs
    s.run("gen idyr = id*100 + yr");
    s.run("reghdfe y x1 x2, absorb(idyr)");
    let b0 = s.scalar("_b[x1]");
    let out = s.run("reghdfe y x1 x2, absorb(i.id#i.yr)");
    assert_close(s.scalar("_b[x1]"), b0, 1e-9);
    assert!(out.contains("id#yr"), "{out}");
}

#[test]
fn heterogeneous_slopes_match_dummy_interactions() {
    let mut s = Session::new();
    panel(&mut s);
    s.run("regress y x1 x2 i.id ibn.id#c.z");
    let b0 = s.scalar("_b[x1]");
    let se0 = s.scalar("_se[x1]");
    let out = s.run("reghdfe y x1 x2, absorb(id##c.z)");
    assert_close(s.scalar("_b[x1]"), b0, 1e-7);
    assert_close(s.scalar("_se[x1]"), se0, 1e-7);
    assert!(out.contains("id#c.z"), "{out}");
    // two-way with slopes on one dimension
    s.run("regress y x1 x2 i.id ibn.id#c.z i.yr");
    let b1 = s.scalar("_b[x1]");
    s.run("reghdfe y x1 x2, absorb(id##c.z yr)");
    assert_close(s.scalar("_b[x1]"), b1, 1e-6);
}

#[test]
fn weights_noabsorb_and_postestimation() {
    let mut s = Session::new();
    panel(&mut s);
    s.run("gen w = 1 + mod(_n, 4)");
    s.run("regress y x1 x2 i.id i.yr [aw=w]");
    let (b0, se0) = coefs_and_ses(&mut s, &["x1", "x2"]);
    s.run("reghdfe y x1 x2 [aw=w], absorb(id yr)");
    let (b, se) = coefs_and_ses(&mut s, &["x1", "x2"]);
    assert_all_close(&b, &b0, 1e-7);
    assert_all_close(&se, &se0, 1e-7);
    s.run("regress y x1 x2 i.id i.yr [fw=w]");
    let (b0, se0) = coefs_and_ses(&mut s, &["x1", "x2"]);
    s.run("reghdfe y x1 x2 [fw=w], absorb(id yr)");
    let (b, se) = coefs_and_ses(&mut s, &["x1", "x2"]);
    assert_all_close(&b, &b0, 1e-7);
    assert_all_close(&se, &se0, 1e-7);
    // noabsorb: plain OLS
    s.run("regress y x1 x2");
    let (b0, se0) = coefs_and_ses(&mut s, &["x1", "x2", "_cons"]);
    s.run("reghdfe y x1 x2, noabsorb");
    let (b, se) = coefs_and_ses(&mut s, &["x1", "x2", "_cons"]);
    assert_all_close(&b, &b0, 1e-9);
    assert_all_close(&se, &se0, 1e-9);
    // tests use the t distribution with e(df_r)
    s.run("reghdfe y x1 x2, absorb(id yr)");
    let out = s.run("lincom x1 + x2");
    let se = s.scalar("r(se)");
    assert!(out.contains("x1 + x2") || out.contains("(1)"), "{out}");
    assert!(se > 0.0);
    assert_eq!(s.scalar("r(df)"), s.scalar("e(df_r)"));
    let out = s.run("test x1 x2");
    assert!(out.contains("F("), "{out}");
    s.run("margins, dydx(x1)");
    s.run("matrix T = r(table)");
    assert_close(s.scalar("T[1,1]"), s.scalar("_b[x1]"), 1e-9);
    // replay
    let out = s.run("reghdfe");
    assert!(out.contains("HDFE Linear regression"));
}

#[test]
fn regressor_collinear_with_fixed_effects_is_omitted() {
    let mut s = Session::new();
    panel(&mut s);
    s.run("gen idx = id/3");
    let out = s.run("reghdfe y x1 idx, absorb(id yr)");
    assert!(out.contains("collinear with the fixed effects"), "{out}");
    assert!(out.contains("(omitted)"), "{out}");
    s.run("regress y x1 i.id i.yr");
    let b0 = s.scalar("_b[x1]");
    s.run("reghdfe y x1 idx, absorb(id yr)");
    assert_close(s.scalar("_b[x1]"), b0, 1e-8);
}

#[test]
fn package_is_reported_as_built_in() {
    let mut s = Session::new();
    let out = s.run("ssc install reghdfe");
    assert!(out.contains("reghdfe"), "{out}");
    assert!(out.contains("nothing to install"), "{out}");
    s.run("ssc install ftools");
    s.run("which reghdfe");
}

#[test]
fn ivreghdfe_matches_two_stage_least_squares_with_dummies() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 800
         set seed 11
         gen id = 1 + floor(runiform()*30)
         gen yr = 1 + floor(runiform()*6)
         gen z1 = rnormal()
         gen z2 = rnormal()
         gen u = rnormal()
         gen x1 = rnormal()
         gen x2 = 0.7*z1 + 0.5*z2 + 0.5*u + id/20 + rnormal()
         gen y = 1 + 0.5*x1 - 0.8*x2 + id/7 + yr/3 + u",
    );
    s.run("ivregress 2sls y x1 (x2 = z1 z2) i.id i.yr, small");
    let (b0, se0) = coefs_and_ses(&mut s, &["x1", "x2"]);
    let out = s.run("ivreghdfe y x1 (x2 = z1 z2), absorb(id yr) small");
    let (b, se) = coefs_and_ses(&mut s, &["x1", "x2"]);
    assert_all_close(&b, &b0, 1e-7);
    assert_all_close(&se, &se0, 1e-7);
    assert!(out.contains("Instrumented:         x2"), "{out}");
    assert!(out.contains("Excluded instruments: z1 z2"), "{out}");
    assert_eq!(s.run("display e(cmd)").trim(), "ivreghdfe");
    // large-sample default: same point estimates, z statistics
    let out = s.run("ivreghdfe y x1 (x2 = z1 z2), absorb(id yr)");
    assert_close(s.scalar("_b[x2]"), b0[1], 1e-7);
    assert!(out.contains("P>|z|"), "{out}");
    let out = s.run("ssc install ivreghdfe");
    assert!(out.contains("nothing to install"), "{out}");
}
