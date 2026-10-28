//! Regression discontinuity (`rdrobust`, `rdbwselect`, `rdplot`,
//! `rddensity`, `rdbwdensity`), binned scatter plots (`binscatter`) and
//! Poisson regression with absorbed fixed effects (`ppmlhdfe`).
//!
//! The RD reference values were produced by an independent implementation
//! of the same estimators on the deterministic design below (no random
//! numbers, so both sides see identical data). Point estimates are also
//! checked against hand-computed weighted local linear fits, and
//! `ppmlhdfe` against `poisson` with explicit dummies.

use ferrostat_tests::*;

/// 800 points on a grid of the running variable, a jump of 0.4 at zero and
/// a deterministic pseudo-noise term.
fn rd_design(s: &mut Session) {
    s.script(
        "clear
         set obs 800
         gen double x = (2*_n - 1 - 800)/800
         gen double e = mod(_n*7919, 1009)/1009 - 0.5
         gen double y = 0.5 + 0.8*x + 0.3*x^2 + 0.4*(x>=0) + e
         gen double z = mod(_n*104729, 997)/997
         gen double y2 = y + 0.5*z
         gen t = ((x>=0) & mod(_n,6)!=0) | ((x<0) & mod(_n,9)==0)
         gen double w = cond(x<0, x, 0.7*x)
         replace w = round(w/0.002)*0.002",
    );
}

#[test]
fn rdrobust_matches_reference_values() {
    let mut s = Session::new();
    rd_design(&mut s);
    let out = s.run("rdrobust y x");
    assert!(out.contains("Sharp RD estimates using local polynomial regression."), "{out}");
    assert!(out.contains("RD Effect"), "{out}");
    assert!(out.contains("BW type       =      mserd"), "{out}");
    assert_close(s.scalar("e(tau_cl)"), 0.38408322, 1e-6);
    assert_close(s.scalar("e(tau_bc)"), 0.3861198, 1e-6);
    assert_close(s.scalar("e(se_tau_cl)"), 0.06854591, 1e-6);
    assert_close(s.scalar("e(se_tau_rb)"), 0.08278503, 1e-6);
    assert_close(s.scalar("e(h_l)"), 0.40595413, 1e-6);
    assert_close(s.scalar("e(b_r)"), 0.61111708, 1e-6);
    assert_eq!(s.scalar("e(N)"), 800.0);
    assert_close(s.scalar("_b[Robust]"), 0.3861198, 1e-6);
    assert_close(s.scalar("_se[Robust]"), 0.08278503, 1e-6);
    assert_eq!(s.run("display e(cmd)").trim(), "rdrobust");
    assert_eq!(s.run("display e(vce_type)").trim(), "NN");
    // covariates, fuzzy design, HC2 with side-specific bandwidths
    s.run("qui rdrobust y2 x, covs(z)");
    assert_close(s.scalar("e(tau_cl)"), 0.38376765, 1e-6);
    assert_close(s.scalar("e(se_tau_rb)"), 0.08277779, 1e-6);
    assert_close(s.scalar("e(h_l)"), 0.40591752, 1e-6);
    s.run("qui rdrobust y x, fuzzy(t)");
    assert_close(s.scalar("e(tau_cl)"), 0.53844191, 1e-6);
    assert_close(s.scalar("e(tau_bc)"), 0.54339381, 1e-6);
    assert_close(s.scalar("e(se_tau_rb)"), 0.1213796, 1e-6);
    assert_close(s.scalar("e(b_l)"), 0.60770485, 1e-6);
    s.run("qui rdrobust y x, vce(hc2) bwselect(msetwo)");
    assert_close(s.scalar("e(tau_cl)"), 0.38426291, 1e-6);
    assert_close(s.scalar("e(se_tau_cl)"), 0.07115542, 1e-6);
    assert_close(s.scalar("e(h_l)"), 0.40117471, 1e-6);
    assert_close(s.scalar("e(h_r)"), 0.40540491, 1e-6);
    assert_close(s.scalar("e(b_r)"), 0.6133322, 1e-6);
}

#[test]
fn rdrobust_conventional_equals_weighted_local_linear_fit() {
    let mut s = Session::new();
    rd_design(&mut s);
    s.run("qui rdrobust y x, h(0.3) kernel(triangular)");
    let tau = s.scalar("e(tau_cl)");
    let l = s.scalar("e(tau_cl_l)");
    let r = s.scalar("e(tau_cl_r)");
    s.script(
        "gen double kw = max(0, 1 - abs(x)/0.3)
         qui regress y x [aw=kw] if x < 0 & kw > 0
         scalar a_l = _b[_cons]
         qui regress y x [aw=kw] if x >= 0 & kw > 0
         scalar a_r = _b[_cons]",
    );
    let (al, ar) = (s.scalar("a_l"), s.scalar("a_r"));
    assert_close(l, al, 1e-9);
    assert_close(r, ar, 1e-9);
    assert_close(tau, ar - al, 1e-9);
    // with h = b and p = 1, q = 2 the bias-corrected estimate is the
    // difference of local quadratic intercepts
    s.run("qui rdrobust y x, h(0.3) b(0.3)");
    let bc = s.scalar("e(tau_bc)");
    s.script(
        "gen double x2 = x^2
         qui regress y x x2 [aw=kw] if x < 0 & kw > 0
         scalar q_l = _b[_cons]
         qui regress y x x2 [aw=kw] if x >= 0 & kw > 0
         scalar q_r = _b[_cons]",
    );
    assert_close(bc, s.scalar("q_r") - s.scalar("q_l"), 1e-8);
}

#[test]
fn rdbwselect_agrees_with_rdrobust() {
    let mut s = Session::new();
    rd_design(&mut s);
    let out = s.run("rdbwselect y x, all");
    for name in ["mserd", "msetwo", "msesum", "msecomb1", "msecomb2", "cerrd", "certwo", "cersum", "cercomb1", "cercomb2"] {
        assert!(out.contains(name), "{name}: {out}");
    }
    let h = s.scalar("e(h_mserd)");
    let b = s.scalar("e(b_mserd)");
    assert_close(h, 0.40595413, 1e-6);
    let cer = s.scalar("e(h_cerrd)");
    assert!(cer < h, "CER bandwidth is smaller");
    s.run("qui rdrobust y x, bwselect(cerrd)");
    assert_close(s.scalar("e(h_l)"), cer, 1e-10);
    assert_close(s.scalar("e(b_l)"), b, 1e-10);
    s.run("qui rdbwselect y x, bwselect(msetwo)");
    assert_close(s.scalar("e(h_msetwo_l)"), {
        s.run("qui rdrobust y x, bwselect(msetwo)");
        s.scalar("e(h_l)")
    }, 1e-10);
}

#[test]
fn rdplot_bins_and_generated_variables() {
    let mut s = Session::new();
    rd_design(&mut s);
    let out = s.run("rdplot y x, genvars hide");
    assert!(out.contains("RD Plot with evenly spaced mimicking variance number of bins"), "{out}");
    assert_eq!(s.scalar("e(J_star_l)"), 30.0);
    assert_eq!(s.scalar("e(J_star_r)"), 53.0);
    assert_eq!(row(&out, "IMSE-optimal bins").unwrap(), vec![7.0, 12.0]);
    assert!(s.has_var("rdplot_mean_y"));
    // bin means recomputed by hand for the first left bin
    s.script(
        "qui summarize y if rdplot_id == -30
         scalar m = r(mean)
         qui summarize rdplot_mean_y if rdplot_id == -30
         scalar g = r(mean)",
    );
    assert_close(s.scalar("g"), s.scalar("m"), 1e-12);
    // the plot is drawn when not hidden
    s.run("rdplot y x, nbins(10 10) ci(95)");
    assert_eq!(s.scalar("e(J_star_l)"), 10.0);
    let f = temp_dir("t36_rdplot").join("rd.svg");
    s.run(&format!("graph export \"{}\", replace", f.display()));
    let svg = std::fs::read_to_string(&f).unwrap();
    assert!(svg.contains("<svg"));
}

#[test]
fn rddensity_manipulation_test() {
    let mut s = Session::new();
    rd_design(&mut s);
    let out = s.run("rddensity w");
    assert!(out.contains("RD Manipulation test using local polynomial density estimation."), "{out}");
    assert!(out.contains("P-values of binomial tests"), "{out}");
    assert_close(s.scalar("e(T_q)"), 1.324597301873422, 1e-7);
    assert_close(s.scalar("e(h_l)"), 0.998, 1e-9);
    assert_close(s.scalar("e(f_ql)"), 0.50073848172601, 1e-8);
    assert_close(s.scalar("e(f_qr)"), 0.7147280821271179, 1e-8);
    assert_close(s.scalar("e(se_q)"), 0.16155068419545723, 1e-8);
    s.run("qui rddensity w, h(0.3 0.25)");
    assert_close(s.scalar("e(T_q)"), 0.7501736132064681, 1e-7);
    assert_close(s.scalar("e(f_qr)"), 0.7126611686069282, 1e-8);
    s.run("qui rddensity w, h(0.3) fitselect(restricted)");
    assert_close(s.scalar("e(T_q)"), 1.2446336421420614, 1e-7);
    s.run("qui rdbwdensity w");
    assert_close(s.scalar("el(e(h), 2, 1)"), 0.7, 1e-9);
    assert_close(s.scalar("el(e(h), 1, 2)"), 2.868388, 1e-5);
}

#[test]
fn ppmlhdfe_equals_poisson_with_dummies() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 600
         set seed 11
         gen id = 1 + floor(runiform()*30)
         gen yr = 1 + floor(runiform()*6)
         gen x1 = rnormal()
         gen x2 = runiform()
         gen y = rpoisson(exp(0.3*x1 - 0.5*x2 + id/30 + yr/10))
         replace y = 0 if id == 3",
    );
    s.run("qui poisson y x1 x2 i.id i.yr if id != 3, vce(robust)");
    let b0 = [s.scalar("_b[x1]"), s.scalar("_b[x2]")];
    let se0 = [s.scalar("_se[x1]"), s.scalar("_se[x2]")];
    let ll0 = s.scalar("e(ll)");
    let out = s.run("ppmlhdfe y x1 x2, absorb(id yr)");
    assert!(out.contains("HDFE PPML regression"), "{out}");
    assert!(out.contains("separated by a fixed effect"), "{out}");
    assert_all_close(&[s.scalar("_b[x1]"), s.scalar("_b[x2]")], &b0, 1e-6);
    assert_all_close(&[s.scalar("_se[x1]"), s.scalar("_se[x2]")], &se0, 1e-5);
    assert_close(s.scalar("e(ll)"), ll0, 1e-6);
    assert_eq!(s.scalar("e(N)"), 578.0);
    assert_eq!(s.run("display e(cmd)").trim(), "ppmlhdfe");
    s.run("qui poisson y x1 x2 i.id i.yr if id != 3, vce(cluster id)");
    let sec = s.scalar("_se[x1]");
    s.run("qui ppmlhdfe y x1 x2, absorb(id yr) vce(cluster id) d(fesum)");
    assert_close(s.scalar("_se[x1]"), sec, 1e-5);
    assert_eq!(s.scalar("e(N_clust)"), 29.0);
    assert!(s.has_var("fesum"));
}

#[test]
fn ppmlhdfe_drops_separated_observations() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 2000
         set seed 5
         gen id = 1 + floor(runiform()*50)
         gen x1 = rnormal()
         gen z = runiform() < 0.1
         gen y = rpoisson(exp(0.3*x1 + id/50))
         replace y = 0 if z == 1",
    );
    s.run("count if z == 1");
    let nz = s.scalar("r(N)");
    let out = s.run("ppmlhdfe y x1 z, absorb(id)");
    assert!(out.contains("ReLU method dropped"), "{out}");
    assert_eq!(s.scalar("e(N)"), 2000.0 - nz);
    s.run("qui poisson y x1 i.id if z == 0, vce(robust)");
    let b = s.scalar("_b[x1]");
    s.run("qui ppmlhdfe y x1 z, absorb(id)");
    assert_close(s.scalar("_b[x1]"), b, 1e-6);
}

#[test]
fn binscatter_bin_means() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 400
         set seed 3
         gen x = rnormal()
         gen g = runiform() < 0.5
         gen y = 1 + 2*x + g + rnormal()",
    );
    let dir = temp_dir("t36_binscatter");
    let base = dir.join("bins");
    s.run(&format!("binscatter y x, nquantiles(10) savedata(\"{}\") replace", base.display()));
    // linear fit coefficients equal OLS on the raw data
    let slope = s.scalar("el(r(coefs), 1, 2)");
    s.run("qui regress y x");
    assert_close(slope, s.scalar("_b[x]"), 1e-9);
    // bin means equal those from deciles of x
    let csv = std::fs::read_to_string(format!("{}.csv", base.display())).unwrap();
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines[0], "x,y");
    assert_eq!(lines.len(), 11);
    s.run("qui summarize x, detail");
    let p10 = s.scalar("r(p10)");
    s.run(&format!("qui summarize y if x <= {p10:.17}"));
    let first: Vec<f64> = lines[1].split(',').map(|v| v.parse().unwrap()).collect();
    assert_close(first[1], s.scalar("r(mean)"), 1e-8);
    // controls are partialled out of both variables, means added back
    s.run("binscatter y x, controls(g) by(g) linetype(qfit)");
    s.run("qui regress y x g");
    let bx = s.scalar("_b[x]");
    s.run("qui binscatter y x, controls(g)");
    assert_close(s.scalar("el(r(coefs), 1, 2)"), bx, 1e-9);
}

#[test]
fn packages_are_reported_native() {
    let mut s = Session::new();
    for pkg in ["rdrobust", "rddensity", "binscatter", "ppmlhdfe"] {
        let out = s.run(&format!("ssc install {pkg}"));
        assert!(out.contains("provided natively"), "{pkg}: {out}");
    }
    let out = s.run("ssc install rdrobust");
    assert!(out.contains("rdbwselect") && out.contains("rdplot"), "{out}");
}
