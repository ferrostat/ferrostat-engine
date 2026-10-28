//! Vector autoregressions and error-correction models: `var`, `svar`,
//! `varsoc`, `varstable`, `varlmar`, `varnorm`, `vargranger`, `varwle`,
//! `irf`, `vec`, `vecrank`, `veclmar`, `vecstable`, `fcast`. Results are
//! checked against independent computations with other commands (OLS by
//! equation, Wald statistics from the coefficient table, closed forms of
//! the impulse responses).

use ferrostat_tests::*;

/// A simulated stationary VAR(1) in three variables, quarterly.
fn data() -> Session {
    let mut s = Session::new();
    s.script(
        "set obs 160
set seed 2718
generate int qtr = tq(1960q1) + _n - 1
format qtr %tq
tsset qtr
generate double e1 = rnormal()
generate double e2 = rnormal()
generate double e3 = rnormal()
generate double y1 = e1
generate double y2 = e2
generate double y3 = e3
replace y1 = 0.5*L.y1 + 0.2*L.y2 + e1 in 2/l
replace y2 = 0.1*L.y1 + 0.4*L.y2 - 0.1*L.y3 + e2 + 0.5*e1 in 2/l
replace y3 = 0.3*L.y2 + 0.3*L.y3 + e3 + 0.3*e2 in 2/l",
    );
    s
}

/// Numbers on the row of an `irf table` for step `h`.
fn step_row(out: &str, h: usize) -> Vec<f64> {
    out.lines()
        .find_map(|l| {
            let rest = l.strip_prefix('|')?;
            let (head, tail) = rest.split_once('|')?;
            (head.trim() == h.to_string()).then(|| numbers(tail))
        })
        .unwrap_or_else(|| panic!("step {h} not found in\n{out}"))
}

#[test]
fn var_equals_equation_by_equation_ols() {
    let mut s = data();
    let out = s.run("var y1 y2 y3, lags(1/2) dfk");
    assert!(out.contains("Vector autoregression") && out.contains("Det(Sigma_ml)") && out.contains("Small-sample"), "{out}");
    let b = s.scalar("_b[y2:L2.y3]");
    let se = s.scalar("_se[y2:L2.y3]");
    let n = s.scalar("e(N)");
    assert_eq!(n, 158.0);
    s.run("regress y2 L.y1 L2.y1 L.y2 L2.y2 L.y3 L2.y3");
    assert_close(b, s.scalar("_b[L2.y3]"), 1e-9);
    // dfk divides Σ by T - m, as regress does
    assert_close(se, s.scalar("_se[L2.y3]"), 1e-7);
    // without dfk the ML divisor is used
    s.run("var y1 y2 y3, lags(1/2)");
    assert_close(s.scalar("_se[y2:L2.y3]"), se * ((158.0 - 7.0) / 158.0f64).sqrt(), 1e-7);
    assert_eq!(s.scalar("e(k_eq)"), 3.0);
    assert_eq!(s.scalar("e(mlag)"), 2.0);
}

#[test]
fn univariate_var_likelihood_matches_regress() {
    let mut s = data();
    s.run("var y1, lags(1/3)");
    let ll = s.scalar("e(ll)");
    s.run("regress y1 L.y1 L2.y1 L3.y1");
    assert_close(ll, s.scalar("e(ll)"), 1e-9);
}

#[test]
fn information_criteria_definitions() {
    let mut s = data();
    s.run("var y1 y2, lags(1/2)");
    let (ll, t) = (s.scalar("e(ll)"), s.scalar("e(N)"));
    let tp = 2.0 * 5.0;
    assert_close(s.scalar("e(aic)"), -2.0 * ll / t + 2.0 * tp / t, 1e-10);
    assert_close(s.scalar("e(sbic)"), -2.0 * ll / t + t.ln() / t * tp, 1e-10);
    assert_close(s.scalar("e(hqic)"), -2.0 * ll / t + 2.0 * t.ln().ln() / t * tp, 1e-10);
    let det = s.scalar("e(detsig_ml)");
    assert_close(ll, -t / 2.0 * (2.0 * ((2.0 * std::f64::consts::PI).ln() + 1.0) + det.ln()), 1e-10);
    assert_close(s.scalar("e(fpe)"), det * ((t + 5.0) / (t - 5.0)).powi(2), 1e-10);
    s.run("var y1 y2, lags(1/2) lutstats");
    assert_close(s.scalar("e(aic)"), det.ln() + 2.0 * 2.0 * 4.0 / t, 1e-10);
}

#[test]
fn varsoc_lr_statistics_from_separate_fits() {
    let mut s = data();
    let out = s.run("varsoc y1 y2 y3, maxlag(3)");
    assert!(out.contains("Lag-order selection criteria") && out.contains("* optimal lag") && out.contains("Exogenous: _cons"), "{out}");
    assert_eq!(s.scalar("r(N)"), 157.0);
    s.run("matrix S = r(stats)");
    let ll1 = s.scalar("S[2,2]");
    let ll2 = s.scalar("S[3,2]");
    assert_close(s.scalar("S[3,3]"), 2.0 * (ll2 - ll1), 1e-9);
    // the same models fit on the common sample
    s.run("var y1 y2 y3 if qtr >= tq(1960q4), lags(1)");
    assert_close(s.scalar("e(ll)"), ll1, 1e-9);
    s.run("var y1 y2 y3 if qtr >= tq(1960q4), lags(1/2)");
    assert_close(s.scalar("e(ll)"), ll2, 1e-9);
    // postestimation form uses the estimation sample
    s.run("varsoc");
    assert_eq!(s.scalar("r(mlag)"), 2.0);
}

#[test]
fn varstable_eigenvalues_of_a_var1() {
    let mut s = data();
    s.run("var y1 y2, lags(1)");
    let a = [s.scalar("_b[y1:L.y1]"), s.scalar("_b[y1:L.y2]"), s.scalar("_b[y2:L.y1]"), s.scalar("_b[y2:L.y2]")];
    let out = s.run("varstable");
    assert!(out.contains("satisfies stability condition"), "{out}");
    let tr = a[0] + a[3];
    let det = a[0] * a[3] - a[1] * a[2];
    let disc = tr * tr - 4.0 * det;
    let moduli = if disc >= 0.0 {
        let r1 = ((tr + disc.sqrt()) / 2.0).abs();
        let r2 = ((tr - disc.sqrt()) / 2.0).abs();
        (r1.max(r2), r1.min(r2))
    } else {
        (det.sqrt(), det.sqrt())
    };
    s.run("matrix M = r(Modulus)");
    assert_close(s.scalar("M[1,1]"), moduli.0, 1e-9);
    assert_close(s.scalar("M[1,2]"), moduli.1, 1e-9);
    s.run("varstable, amat(C)");
    assert_close(s.scalar("C[1,2]"), a[1], 1e-12);
}

#[test]
fn granger_and_lag_exclusion_wald_tests() {
    let mut s = data();
    s.run("var y1 y2 y3, lags(1)");
    let z = s.scalar("_b[y1:L.y3]") / s.scalar("_se[y1:L.y3]");
    let out = s.run("vargranger");
    assert!(out.contains("Granger causality Wald tests") && out.contains("ALL"), "{out}");
    s.run("matrix G = r(gstats)");
    // equation y1: rows are y2, y3, ALL
    assert_close(s.scalar("G[2,1]"), z * z, 1e-8);
    assert_eq!(s.scalar("G[3,2]"), 2.0);
    s.run("varwle");
    s.run("matrix W = r(chi2)");
    s.run("matrix D = r(df)");
    assert_eq!(s.scalar("D[1,1]"), 3.0);
    assert_eq!(s.scalar("D[4,1]"), 9.0);
    // small: F statistics with T - m residual degrees of freedom
    s.run("var y1 y2 y3, lags(1) small");
    s.run("vargranger");
    s.run("matrix G = r(gstats)");
    assert_close(s.scalar("G[2,1]"), z * z, 1e-8);
    assert_eq!(s.scalar("G[2,3]"), 159.0 - 4.0);
}

#[test]
fn lm_and_normality_tests_run() {
    let mut s = data();
    s.run("var y1 y2 y3, lags(1/2)");
    let out = s.run("varlmar, mlag(3)");
    assert!(out.contains("Lagrange-multiplier test") && out.contains("H0: no autocorrelation"), "{out}");
    s.run("matrix LM = r(lm)");
    assert_eq!(s.scalar("LM[3,2]"), 9.0);
    let p = s.scalar("LM[1,3]");
    assert!(p > 0.0 && p < 1.0);
    let out = s.run("varnorm");
    assert!(out.contains("Jarque-Bera test") && out.contains("Skewness test") && out.contains("Kurtosis test"), "{out}");
    s.run("matrix J = r(jb)");
    s.run("matrix SK = r(skewness)");
    s.run("matrix KU = r(kurtosis)");
    // Jarque-Bera = skewness + kurtosis statistics
    assert_close(s.scalar("J[4,1]"), s.scalar("SK[4,1]") + s.scalar("KU[4,1]"), 1e-9);
    assert_eq!(s.scalar("J[4,2]"), 6.0);
}

#[test]
fn irf_closed_forms_and_delta_method() {
    let mut s = data();
    let dir = temp_dir("irf_closed_forms");
    s.run(&format!("cd \"{}\"", dir.display()));
    s.run("var y1 y2, lags(1)");
    let a12 = s.scalar("_b[y1:L.y2]");
    let se12 = s.scalar("_se[y1:L.y2]");
    let a11 = s.scalar("_b[y1:L.y1]");
    let a21 = s.scalar("_b[y2:L.y1]");
    s.run("matrix S = e(Sigma)");
    let (s11, s21) = (s.scalar("S[1,1]"), s.scalar("S[2,1]"));
    let out = s.run("irf create base, set(myirfs) step(4)");
    assert!(out.contains("myirfs.irf created") && out.contains("now active") && out.contains("updated"), "{out}");
    // irf at step 1 is the coefficient, with its standard error
    let out = s.run("irf table irf, impulse(y2) response(y1) stderror noci");
    assert!(out.contains("irfname = base, impulse = y2, and response = y1"), "{out}");
    let r1 = step_row(&out, 1);
    assert_close(r1[0], a12, 1e-5);
    assert_close(r1[1], se12, 1e-4);
    // step 2: (A²)₁₂ = a11 a12 + a12 a22
    let a22 = s.scalar("_b[y2:L.y2]");
    assert_close(step_row(&out, 2)[0], a11 * a12 + a12 * a22, 1e-5);
    // orthogonalized response at step 0 is the Cholesky factor
    let out = s.run("irf table oirf, impulse(y1) response(y2) noci");
    assert_close(step_row(&out, 0)[0], s21 / s11.sqrt(), 1e-5);
    // cumulative irf adds the steps
    let out = s.run("irf table cirf, impulse(y1) response(y2) noci");
    assert_close(step_row(&out, 1)[0], a21, 1e-5);
    // first variable in the ordering explains all of its own 1-step variance
    let out = s.run("irf table fevd, impulse(y1) response(y1) noci");
    assert_close(step_row(&out, 1)[0], 1.0, 1e-9);
    // a second ordering, describe, rename and drop
    s.run("irf create other, order(y2 y1) step(4)");
    let out = s.run("irf table fevd, impulse(y2) response(y2) noci irf(other)");
    assert_close(step_row(&out, 1)[0], 1.0, 1e-9);
    let out = s.run("irf describe");
    assert!(out.contains("base") && out.contains("other") && out.contains("y2 y1"), "{out}");
    s.run("irf rename other second");
    s.run("irf drop base");
    let out = s.run("irf describe");
    assert!(out.contains("second") && !out.contains("base "), "{out}");
    s.fails("irf create second");
    s.run("irf create second, replace step(2)");
    // graphs
    s.run("irf graph oirf fevd, impulse(y1) response(y2)");
    let f = dir.join("irf.svg");
    s.run(&format!("graph export \"{}\", replace", f.display()));
    assert!(std::fs::metadata(&f).map(|m| m.len()).unwrap_or(0) > 1000);
    s.run("irf graph irf");
    s.run("irf ograph (second y1 y2 oirf) (second y2 y1 oirf)");
    s.run("irf cgraph (second y1 y2 oirf) (second y2 y1 irf)");
    let out = s.run("irf ctable (second y1 y2 oirf) (second y2 y2 fevd), stderror");
    assert!(out.contains("(2) irfname = second, impulse = y2, and response = y2."), "{out}");
}

#[test]
fn irf_bootstrap_and_dynamic_multipliers() {
    let mut s = data();
    let dir = temp_dir("irf_bootstrap");
    s.run(&format!("cd \"{}\"", dir.display()));
    s.run("var y1 y2, lags(1/2) exog(y3 L.y3)");
    let b0 = s.scalar("_b[y1:y3]");
    let b1 = s.scalar("_b[y1:L.y3]");
    let a11 = s.scalar("_b[y1:L.y1]");
    let a12 = s.scalar("_b[y1:L.y2]");
    let c0 = s.scalar("_b[y2:y3]");
    s.run("irf set dmfile");
    s.run("irf create dm, step(3)");
    let out = s.run("irf table dm, impulse(y3) response(y1) noci");
    assert_close(step_row(&out, 0)[0], b0, 1e-5);
    assert_close(step_row(&out, 1)[0], b1 + a11 * b0 + a12 * c0, 1e-5);
    s.run("set seed 1");
    s.run("var y1 y2, lags(1)");
    s.run("irf create boot, step(3) bs reps(50)");
    let out = s.run("irf table irf, impulse(y2) response(y1) stderror noci irf(boot)");
    let se = step_row(&out, 1)[1];
    assert!(se > 0.0 && se < 1.0, "{out}");
}

#[test]
fn svar_exactly_identified_recovers_sigma() {
    let mut s = data();
    s.run("matrix A = (1,0\\.,1)");
    s.run("matrix B = (.,0\\0,.)");
    let out = s.run("svar y1 y2, aeq(A) beq(B)");
    assert!(out.contains("Exactly identified model") && out.contains("(constrained)") && out.contains("[/A]1_1 = 1"), "{out}");
    let ll = s.scalar("e(ll)");
    s.run("matrix Ah = e(A)");
    s.run("matrix Bh = e(B)");
    s.run("matrix S = e(Sigma)");
    let (s11, s21, s22) = (s.scalar("S[1,1]"), s.scalar("S[2,1]"), s.scalar("S[2,2]"));
    // A Σ A' = B B'
    let a21 = s.scalar("Ah[2,1]");
    assert_close(a21, -s21 / s11, 1e-6);
    assert_close(s.scalar("Bh[1,1]"), s11.sqrt(), 1e-6);
    assert_close(s.scalar("Bh[2,2]"), (s22 - s21 * s21 / s11).sqrt(), 1e-6);
    s.run("var y1 y2");
    assert_close(ll, s.scalar("e(ll)"), 1e-8);
    // structural irf after svar
    let dir = temp_dir("svar_irf");
    s.run(&format!("cd \"{}\"", dir.display()));
    s.run("svar y1 y2, aeq(A) beq(B)");
    s.run("irf create sr, set(sv) step(4)");
    let out = s.run("irf table sirf oirf, impulse(y1) response(y2) noci");
    let r0 = step_row(&out, 0);
    assert_close(r0[0], r0[1], 1e-6);
    s.run("varstable");
    s.run("varwle");
    // over-identified: B diagonal and A diagonal
    s.run("matrix A2 = (1,0\\0,1)");
    let out = s.run("svar y1 y2, aeq(A2) beq(B)");
    assert!(out.contains("Overidentified model") && out.contains("LR test of identifying restrictions"), "{out}");
}

/// Two cointegrated I(1) series and a third random walk.
fn coint() -> Session {
    let mut s = Session::new();
    s.script(
        "set obs 200
set seed 99
generate int t = _n
tsset t
generate double w = sum(rnormal())
generate double x = w + rnormal()
generate double y = 0.5 + 2*w + rnormal()
generate double z = sum(rnormal())",
    );
    s
}

#[test]
fn vecrank_and_vec_likelihoods_agree() {
    let mut s = coint();
    let out = s.run("vecrank x y z, lags(2)");
    assert!(out.contains("Johansen tests for cointegration") && out.contains("Trend: Constant") && out.contains("* selected rank"), "{out}");
    s.run("matrix LL = e(ll)");
    s.run("matrix TR = e(trace)");
    let (ll0, ll1, ll3) = (s.scalar("LL[1,1]"), s.scalar("LL[1,2]"), s.scalar("LL[1,4]"));
    assert_close(s.scalar("TR[1,1]"), 2.0 * (ll3 - ll0), 1e-9);
    assert_close(s.scalar("TR[1,2]"), 2.0 * (ll3 - ll1), 1e-9);
    // one cointegrating relation is found
    assert_eq!(s.scalar("e(k_ce95)"), 1.0);
    s.run("vecrank x y z, lags(2) max ic levela notrace");
    let out = s.run("vec x y z, lags(2) rank(1)");
    assert!(out.contains("Vector error-correction model") && out.contains("Johansen normalization restriction imposed"), "{out}");
    assert_close(s.scalar("e(ll)"), ll1, 1e-9);
    s.run("matrix b = e(beta)");
    assert_eq!(s.scalar("b[1,1]"), 1.0);
    // y ≈ 2x, so β ≈ (1, -0.5, 0)
    assert!((s.scalar("b[1,2]") + 0.5).abs() < 0.05);
    // short-run coefficients: the error-correction term of each equation
    let a = s.scalar("_b[D_x:L._ce1]");
    assert!(a.is_finite());
    let out = s.run("vecstable");
    assert!(out.contains("imposes 2 unit moduli"), "{out}");
    s.run("veclmar");
    s.run("vecnorm");
    for t in ["none", "rconstant", "rtrend", "trend"] {
        s.run(&format!("vec x y z, lags(2) rank(1) trend({t})"));
        s.run(&format!("vecrank x y z, lags(2) trend({t})"));
    }
}

#[test]
fn vec_irf_and_forecast() {
    let mut s = coint();
    let dir = temp_dir("vec_irf");
    s.run(&format!("cd \"{}\"", dir.display()));
    s.run("vec x y, lags(2) rank(1)");
    s.run("irf create v, set(vecirf) step(20)");
    let out = s.run("irf table oirf, impulse(x) response(y)");
    assert!(!out.contains("Lower"), "no standard errors after vec: {out}");
    // permanent effect: the response does not die out
    assert!(step_row(&out, 20)[0].abs() > 1e-3);
    let n = s.n();
    s.run("fcast compute f_, step(5)");
    assert_eq!(s.n(), n + 5);
    let fx = s.nums("f_x");
    let se = s.nums("f_x_SE");
    assert!(fx[n].is_finite() && se[n] > 0.0 && se[n + 4] > se[n]);
    s.run("fcast graph f_x f_y, observed");
}

#[test]
fn fcast_one_step_matches_coefficients() {
    let mut s = data();
    s.run("var y1 y2 if qtr < tq(1995q1), lags(1)");
    let (c, b11, b12) = (s.scalar("_b[y1:_cons]"), s.scalar("_b[y1:L.y1]"), s.scalar("_b[y1:L.y2]"));
    s.run("matrix S = e(Sigma)");
    let s11 = s.scalar("S[1,1]");
    s.run("fcast compute p_, step(3)");
    let y1 = s.nums("y1");
    let y2 = s.nums("y2");
    let f = s.nums("p_y1");
    let se = s.nums("p_y1_SE");
    let qtr = s.nums("qtr");
    let i = qtr.iter().position(|q| *q == 140.0).unwrap(); // 1995q1
    assert_close(f[i], c + b11 * y1[i - 1] + b12 * y2[i - 1], 1e-9);
    // MSE plus estimation uncertainty
    assert!(se[i] > s11.sqrt() && se[i] < 1.2 * s11.sqrt());
    assert!(f[i - 1].is_nan() && f[i + 3].is_nan());
    s.fails("fcast compute p_, step(3)");
    s.run("fcast compute p_, step(3) replace dynamic(tq(1994q3))");
    s.run("fcast graph p_y1, observed");
    s.run("fcast compute q_, step(2) nose");
    assert!(!s.has_var("q_y1_SE"));
}

#[test]
fn invalid_specifications_fail() {
    let mut s = data();
    s.fails("varstable");
    s.fails("var y1, lags(0)");
    s.fails("irf create x");
    s.fails("vec y1 y2, rank(2)");
    let mut s = Session::new();
    s.script("set obs 10\ngenerate x = _n\ngenerate y = x^2");
    s.fails("var x y");
}
