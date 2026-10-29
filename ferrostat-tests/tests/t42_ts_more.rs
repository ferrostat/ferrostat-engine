//! More time-series commands: arch, mgarch, arfima, sspace, ucm, dfactor,
//! mswitch, threshold, forecast, lpirf, structural-break tests and the
//! tsfill/tsappend/tsreport/tsrevar/psdensity utilities. Reference values
//! come from simulated data with known parameters, from equivalent models
//! fitted by other commands, or from closed forms.

use ferrostat_tests::*;

/// GARCH(1,1) errors: h_t = .1 + .1 e_{t-1}^2 + .8 h_{t-1}, mean 0.5.
fn garch_data(n: usize) -> Session {
    let mut s = Session::new();
    s.script(&format!(
        "set obs {n}\nset seed 4242\ngen t = _n\ntsset t\ngen z = rnormal()\n\
         gen double h = 1 in 1\nreplace h = .1 + .1*h[_n-1]*z[_n-1]^2 + .8*h[_n-1] in 2/l\n\
         gen double y = 0.5 + sqrt(h)*z"
    ));
    s
}

/// AR(1) series y_t = 0.6 y_{t-1} + e_t and an exogenous x.
fn ar_data() -> Session {
    let mut s = Session::new();
    s.script(
        "set obs 300\nset seed 777\ngen t = _n\ntsset t\ngen e = rnormal()\ngen double y = e in 1\n\
         replace y = 0.6*y[_n-1] + e in 2/l\ngen x = rnormal()",
    );
    s
}

#[test]
fn garch11_recovers_parameters() {
    let mut s = garch_data(3000);
    let out = s.run("arch y, arch(1) garch(1)");
    assert!(out.contains("ARCH family regression") && out.contains("ARCH") && out.contains("garch"), "{out}");
    assert_close(s.scalar("_b[y:_cons]"), 0.5, 0.05);
    assert_close(s.scalar("_b[ARCH:L.arch]"), 0.1, 0.05);
    assert_close(s.scalar("_b[ARCH:L.garch]"), 0.8, 0.08);
    assert_close(s.scalar("e(N)"), 3000.0, 0.0);
    // conditional variance prediction follows the recursion
    s.run("predict double v, variance");
    s.run("predict double r, residuals");
    let (a0, a1, g1) = (s.scalar("_b[ARCH:_cons]"), s.scalar("_b[ARCH:L.arch]"), s.scalar("_b[ARCH:L.garch]"));
    let v = s.nums("v");
    let r = s.nums("r");
    for t in 1..20 {
        assert_close(v[t], a0 + a1 * r[t - 1] * r[t - 1] + g1 * v[t - 1], 1e-8);
    }
    // variants run and keep a sensible likelihood
    let ll = s.scalar("e(ll)");
    s.run("quietly arch y, arch(1) garch(1) tarch(1)");
    assert!(s.scalar("e(ll)") >= ll - 1e-3);
    s.run("quietly arch y, earch(1) egarch(1)");
    assert!(s.scalar("e(ll)").is_finite());
    s.run("quietly arch y, arch(1) garch(1) distribution(t)");
    assert!(s.scalar("e(tdf)") > 2.0);
}

#[test]
fn sspace_ar1_equals_arima() {
    let mut s = ar_data();
    s.run("quietly arima y, ar(1) noconstant");
    let (ll, rho, sig) = (s.scalar("e(ll)"), s.scalar("_b[ARMA:L.ar]"), s.scalar("_b[sigma:_cons]"));
    s.run("constraint 1 [y]z = 1");
    let out = s.run("sspace (z L.z, state noconstant) (y z, noerror noconstant), constraints(1)");
    assert!(out.contains("State-space model") && out.contains("(constrained)"), "{out}");
    assert_close(s.scalar("e(ll)"), ll, 1e-6);
    assert_close(s.scalar("_b[z:L.z]"), rho, 1e-3);
    assert_close(s.scalar("_b[/state:var(z)]"), sig * sig, 1e-3);
}

#[test]
fn arfima_short_memory_equals_arima_and_d_is_estimated() {
    let mut s = ar_data();
    s.run("quietly arima y, ar(1) ma(1)");
    let ll = s.scalar("e(ll)");
    s.run("quietly arfima y, ar(1) ma(1) smemory");
    assert_close(s.scalar("e(ll)"), ll, 1e-6);
    let out = s.run("arfima y");
    assert!(out.contains("ARFIMA regression") && out.contains(" d |"), "{out}");
    // an AR(1) with rho .6 looks like positive fractional integration
    assert!(s.scalar("_b[ARFIMA:d]") > 0.1);
    let soc = s.run("arimasoc y, maxar(1) maxma(1)");
    assert!(soc.contains("(1,0)") && soc.contains('*'), "{soc}");
}

#[test]
fn psdensity_matches_ar1_formula() {
    let mut s = ar_data();
    s.run("quietly arima y, ar(1) noconstant");
    let rho = s.scalar("_b[ARMA:L.ar]");
    s.run("psdensity double sd double w");
    let (sd, w) = (s.nums("sd"), s.nums("w"));
    assert_close(w[0], 0.0, 1e-12);
    assert_close(*w.last().unwrap(), std::f64::consts::PI, 1e-12);
    // normalized density of an AR(1): (1 - rho^2) / (2 pi (1 - 2 rho cos w + rho^2))
    for i in [0, 50, 299] {
        let expect = (1.0 - rho * rho) / (2.0 * std::f64::consts::PI * (1.0 - 2.0 * rho * w[i].cos() + rho * rho));
        assert_close(sd[i], expect, 1e-3);
    }
}

#[test]
fn tsfill_tsappend_tsreport_tsrevar() {
    let mut s = Session::new();
    s.script("set obs 10\ngen t = _n\ndrop in 4/5\ngen y = t*2\ntsset t");
    let out = s.run("tsreport");
    assert!(out.contains("Number of gaps in sample: 1"), "{out}");
    assert_close(s.scalar("r(N_gaps)"), 1.0, 0.0);
    s.run("tsfill");
    assert_eq!(s.n(), 10);
    assert_all_close(&s.nums("t"), &(1..=10).map(|v| v as f64).collect::<Vec<_>>(), 0.0);
    assert!(s.nums("y")[3].is_nan());
    s.run("tsappend, add(5)");
    assert_eq!(s.n(), 15);
    assert_close(s.nums("t")[14], 15.0, 0.0);
    s.run("tsappend, last(17)");
    assert_eq!(s.n(), 17);
    s.run("tsrevar L.y D.y y");
    let vl = s.run("display \"`r(varlist)'\"");
    let names: Vec<&str> = vl.split_whitespace().collect();
    assert_eq!(names.len(), 3);
    assert_eq!(names[2], "y");
    let lag = s.nums(names[0]);
    assert_close(lag[1], 2.0, 0.0);
    // panels: tsfill, full
    let mut p = Session::new();
    p.script("set obs 4\ngen id = cond(_n<=2,1,2)\ngen t = cond(_n==1,1,cond(_n==2,3,cond(_n==3,2,5)))\nxtset id t\ntsfill, full");
    assert_eq!(p.n(), 10);
}

#[test]
fn mswitch_dr_recovers_regime_means() {
    let mut s = Session::new();
    s.script(
        "set obs 400\nset seed 99\ngen t = _n\ntsset t\ngen u = runiform()\ngen s = 1 in 1\n\
         replace s = cond(s[_n-1]==1, cond(u<.95,1,2), cond(u<.9,2,1)) in 2/l\n\
         gen y = cond(s==1, 0, 4) + rnormal()",
    );
    let out = s.run("mswitch dr y");
    assert!(out.contains("Markov-switching dynamic regression") && out.contains("p11"), "{out}");
    assert_close(s.scalar("_b[State1:_cons]"), 0.0, 0.2);
    assert_close(s.scalar("_b[State2:_cons]"), 4.0, 0.1);
    assert_close(s.scalar("exp(_b[lnsigma:_cons])"), 1.0, 0.1);
    let p = s.run("estat transition");
    assert!(p.contains("p11") && p.contains("p22"), "{p}");
    s.run("predict pr*, pr");
    let pr1 = s.nums("pr1");
    let pr2 = s.nums("pr2");
    assert_close(pr1[10] + pr2[10], 1.0, 1e-9);
    s.run("quietly mswitch ar y, ar(1)");
    assert!(s.scalar("e(ll)").is_finite());
}

#[test]
fn threshold_finds_the_break_in_the_threshold_variable() {
    let mut s = Session::new();
    s.script("set obs 300\nset seed 5\ngen t = _n\ntsset t\ngen q = runiform()*10\ngen x = rnormal()\ngen y = cond(q<=6, 1 + x, 3 - x) + 0.3*rnormal()");
    let out = s.run("threshold y, regionvars(x) threshvar(q)");
    assert!(out.contains("Threshold regression") && out.contains("Region2"), "{out}");
    let th = s.run("matrix list e(thresholds)");
    let thr = numbers(&th);
    assert!(thr.iter().any(|v| (v - 6.0).abs() < 0.2), "{th}");
    assert_close(s.scalar("_b[Region1:x]"), 1.0, 0.1);
    assert_close(s.scalar("_b[Region2:x]"), -1.0, 0.1);
    assert_close(s.scalar("e(nthresholds)"), 1.0, 0.0);
}

#[test]
fn forecast_solves_dynamically() {
    let mut s = ar_data();
    s.run("quietly regress y L.y");
    let (b1, b0) = (s.scalar("_b[L.y]"), s.scalar("_b[_cons]"));
    s.run("estimates store ar1");
    s.run("tsappend, add(5)");
    s.run("forecast create m1");
    s.run("forecast estimates ar1");
    s.run("forecast identity y2 = 2*y, generate");
    let d = s.run("forecast describe");
    assert!(d.contains("2 endogenous"), "{d}");
    let out = s.run("forecast solve");
    assert!(out.contains("spanning 5 periods"), "{out}");
    let y = s.nums("y");
    let f = s.nums("f_y");
    let f2 = s.nums("f_y2");
    let mut prev = y[299];
    for t in 300..305 {
        let expect = b0 + b1 * prev;
        assert_close(f[t], expect, 1e-9);
        assert_close(f2[t], 2.0 * expect, 1e-9);
        prev = expect;
    }
    s.run("forecast drop");
    assert!(!s.has_var("f_y"));
    s.run("forecast clear");
    s.fails("forecast solve");
}

#[test]
fn lpirf_step_one_is_the_var_coefficient() {
    let mut s = ar_data();
    s.run("quietly regress y L.y L2.y L.x L2.x, vce(robust)");
    let (b, se) = (s.scalar("_b[L.y]"), s.scalar("_se[L.y]"));
    let out = s.run("lpirf y x, steps(3)");
    assert!(out.contains("Local-projection"), "{out}");
    assert_close(s.scalar("_b[y:F1.y]"), b, 1e-9);
    assert_close(s.scalar("_se[y:F1.y]"), se, 1e-6);
}

#[test]
fn structural_break_tests() {
    let mut s = ar_data();
    s.script("replace y = y + 2 in 151/l\nquietly regress y x");
    let out = s.run("estat sbknown, break(151)");
    assert!(out.contains("Known break date"), "{out}");
    let chi2 = s.scalar("r(chi2)");
    // equals the Wald test on the interacted regression
    s.script("gen d = t>=151\ngen dx = d*x\nquietly regress y x d dx\nquietly test d dx");
    assert_close(chi2, s.scalar("r(F)") * 2.0, 1e-6);
    s.run("quietly regress y x");
    let out = s.run("estat sbsingle");
    let est = numbers(out.lines().find(|l| l.contains("Estimated break date")).unwrap())[0];
    assert!((est - 151.0).abs() <= 2.0, "{out}");
    assert!(s.scalar("r(p)") < 0.01);
    let out = s.run("estat sbcusum");
    assert!(out.contains("Recursive") && s.scalar("r(statistic)") > 1.143, "{out}");
}

#[test]
fn ucm_and_dfactor_run() {
    let mut s = Session::new();
    s.script(
        "set obs 200\nset seed 31\ngen t = _n\ntsset t\ngen double mu = rnormal()*0.5 in 1\n\
         replace mu = mu[_n-1] + 0.5*rnormal() in 2/l\ngen double y = mu + rnormal()",
    );
    let out = s.run("ucm y, model(llevel)");
    assert!(out.contains("Unobserved-components model") && out.contains("var(level)"), "{out}");
    assert_close(s.scalar("_b[/var(level)]"), 0.25, 0.8);
    assert_close(s.scalar("_b[/var(y)]"), 1.0, 0.4);
    s.run("predict double lev, states");
    assert!(s.nums("lev")[100].is_finite());
    s.script("gen double f = rnormal() in 1\nreplace f = 0.7*f[_n-1] + rnormal() in 2/l\ngen double a = f + 0.5*rnormal()\ngen double b = 0.5*f + 0.5*rnormal()");
    let out = s.run("dfactor (a b = , noconstant) (f1 = , ar(1))");
    assert!(out.contains("Dynamic-factor model"), "{out}");
    assert_close(s.scalar("_b[f1:L.f1]"), 0.7, 0.15);
}

#[test]
fn mgarch_ccc_recovers_correlation() {
    let mut s = Session::new();
    s.script(
        "set obs 1500\nset seed 8\ngen t = _n\ntsset t\ngen z1 = rnormal()\ngen z2 = 0.5*z1 + sqrt(.75)*rnormal()\n\
         gen double h1 = 1 in 1\nreplace h1 = .1 + .1*h1[_n-1]*z1[_n-1]^2 + .8*h1[_n-1] in 2/l\n\
         gen double h2 = 1 in 1\nreplace h2 = .2 + .05*h2[_n-1]*z2[_n-1]^2 + .8*h2[_n-1] in 2/l\n\
         gen double y1 = sqrt(h1)*z1\ngen double y2 = sqrt(h2)*z2",
    );
    let out = s.run("mgarch ccc (y1 y2 = , noconstant), arch(1) garch(1)");
    assert!(out.contains("Constant conditional correlation"), "{out}");
    assert_close(s.scalar("_b[/Correlation:corr(y1,y2)]"), 0.5, 0.06);
    s.run("quietly mgarch dcc (y1 y2 = , noconstant), arch(1) garch(1)");
    assert!(s.scalar("_b[/Adjustment:lambda2]") >= 0.0);
}
