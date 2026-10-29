//! Remaining base-reference commands: truncated and zero-inflated ordered
//! models, stereotype logit, suest, fractional polynomials, z tests,
//! symmetry, stem-and-leaf, vwls, ratio, tetrachoric, dydx/integ, fvset,
//! fvrevar, rwgen and the prefix tools. Checks use published worked
//! examples (data typed in), closed forms, or equalities with existing
//! estimators.

use ferrostat_tests::*;

/// Deterministic regression data.
fn data(s: &mut Session, n: usize) {
    s.script(&format!(
        "set obs {n}
         gen double x1 = mod(_n*13, 17)/4 - 2
         gen double x2 = mod(_n*7, 11)/3
         gen double u = (mod(_n*53, 97) + 0.5)/97
         gen double e = invnormal(u)
         gen double y = 1 + 0.5*x1 - 0.3*x2 + e
         gen double cnt = 1 + floor(exp(0.4 + 0.2*x1) * (1 + u))
         gen byte d = (y > 1.2)
         gen g = mod(_n, 3) + 1"
    ));
}

#[test]
fn truncreg_without_limits_is_ml_regression() {
    let mut s = Session::new();
    data(&mut s, 120);
    s.run("regress y x1 x2");
    let (b1, b2, rss) = (s.scalar("_b[x1]"), s.scalar("_b[x2]"), s.scalar("e(rss)"));
    let out = s.run("truncreg y x1 x2");
    assert!(out.contains("Truncated regression"), "{out}");
    assert_close(s.scalar("_b[x1]"), b1, 1e-6);
    assert_close(s.scalar("_b[x2]"), b2, 1e-6);
    assert_close(s.scalar("e(sigma)"), (rss / 120.0).sqrt(), 1e-5);
    // truncation drops the observations below the limit
    let out = s.run("truncreg y x1 x2, ll(0)");
    let ntr = s.scalar("e(N_bf)");
    assert!(ntr > 0.0 && out.contains("obs truncated"), "{out}");
    assert_close(s.scalar("e(N)") + ntr, 120.0, 0.0);
    s.run("predict double ec, e");
    assert!(s.has_var("ec"));
}

#[test]
fn tpoisson_tnbreg_reduce_to_untruncated_models() {
    let mut s = Session::new();
    data(&mut s, 150);
    s.run("poisson cnt x1 x2");
    let (b1, ll) = (s.scalar("_b[x1]"), s.scalar("e(ll)"));
    // a limit below zero truncates nothing
    s.run("tpoisson cnt x1 x2, ll(-1)");
    assert_close(s.scalar("_b[x1]"), b1, 1e-6);
    assert_close(s.scalar("e(ll)"), ll, 1e-6);
    // zero truncation: y >= 1 everywhere here, the likelihood is larger
    let out = s.run("tpoisson cnt x1 x2");
    assert!(out.contains("Truncated Poisson regression") && out.contains("Limits:  lower ="), "{out}");
    assert!(s.scalar("e(ll)") > ll);
    s.run("predict double cm, cm");
    s.run("predict double nn, n");
    s.run("gen double ratio = cm/nn");
    s.run("summarize ratio");
    assert!(s.scalar("r(min)") >= 1.0);
    s.run("nbreg cnt x1 x2");
    let llnb = s.scalar("e(ll)");
    s.run("tnbreg cnt x1 x2, ll(-1)");
    assert_close(s.scalar("e(ll)"), llnb, 1e-4);
    let out = s.run("tnbreg cnt x1 x2");
    assert!(out.contains("Truncated negative binomial regression") && out.contains("Dispersion: mean"), "{out}");
}

#[test]
fn slogit_full_dimension_equals_mlogit() {
    let mut s = Session::new();
    data(&mut s, 200);
    s.run("gen byte r3 = cond(y < 0.5, 1, cond(y < 1.5, 2, 3))");
    s.run("mlogit r3 x1 x2, baseoutcome(3)");
    let ll = s.scalar("e(ll)");
    let b = s.scalar("_b[1:x1]");
    let out = s.run("slogit r3 x1 x2, dimension(2)");
    assert!(out.contains("Stereotype logistic regression") && out.contains("(base outcome)"), "{out}");
    assert_close(s.scalar("e(ll)"), ll, 1e-6);
    // ln Pr(1)/Pr(3) = theta1 - x'beta_1: slopes change sign
    assert_close(s.scalar("_b[dim1:x1]"), -b, 1e-4);
    // the one-dimensional model is nested
    s.run("slogit r3 x1 x2");
    assert!(s.scalar("e(ll)") <= ll + 1e-8);
    s.run("predict p*");
    s.run("gen double tot = p1 + p2 + p3");
    s.run("summarize tot");
    assert_close(s.scalar("r(mean)"), 1.0, 1e-10);
}

#[test]
fn zero_inflated_ordered_models() {
    let mut s = Session::new();
    data(&mut s, 300);
    s.run("gen byte r = cond(y < 0.3, 0, cond(y < 1.2, 1, 2))");
    s.run("replace r = 0 if mod(_n, 7) == 0");
    s.run("ologit r x1 x2");
    let llo = s.scalar("e(ll)");
    let out = s.run("ziologit r x1 x2, inflate(x2)");
    assert!(out.contains("Zero-inflated ordered logit regression") && out.contains("/cut2"), "{out}");
    assert!(s.scalar("e(ll)") >= llo - 1e-6);
    s.run("predict q*");
    s.run("gen double tot = q1 + q2 + q3");
    s.run("summarize tot");
    assert_close(s.scalar("r(mean)"), 1.0, 1e-10);
    let out = s.run("ziologit, or");
    assert!(out.contains("Odds ratio"), "{out}");
    s.run("oprobit r x1 x2");
    let llp = s.scalar("e(ll)");
    s.run("zioprobit r x1 x2, inflate(x2)");
    assert!(s.scalar("e(ll)") >= llp - 1e-6);
}

#[test]
fn suest_of_one_logit_is_robust_logit() {
    let mut s = Session::new();
    data(&mut s, 200);
    s.run("logit d x1 x2, vce(robust)");
    let se = s.scalar("_se[x1]");
    s.run("logit d x1 x2");
    s.run("estimates store m1");
    s.run("regress y x1");
    s.run("estimates store m2");
    let out = s.run("suest m1 m2");
    assert!(out.contains("Simultaneous results for m1, m2"), "{out}");
    assert_close(s.scalar("_se[m1:x1]"), se, 1e-6);
    // regress enters with its mean and log-variance equations
    assert_close(s.scalar("_b[m2_lnvar:_cons]").exp() * 200.0, {
        s.run("quietly regress y x1");
        s.scalar("e(rss)")
    }, 1e-6);
    s.run("suest m1 m2");
    s.run("test [m1]x1 = 0");
    assert!(s.scalar("r(chi2)") > 0.0);
}

#[test]
fn fp_recovers_power_and_fixed_powers_match_regress() {
    let mut s = Session::new();
    data(&mut s, 150);
    s.run("gen double w = 1 + mod(_n*17, 29)/7");
    s.run("gen double yy = 2 + 3/w + 0.05*e");
    let out = s.run("fp <w>: regress yy <w> x1");
    assert!(out.contains("Fractional polynomial comparisons:"), "{out}");
    assert!(out.contains("(fitting 44 models)"), "{out}");
    assert!(s.has_var("w_1"));
    // fixed powers: same as regressing on the transformed variable
    s.run("fp <w>, fp(-1) replace: regress yy <w> x1");
    let b = s.scalar("_b[w_1]");
    s.run("gen double winv = 1/w");
    s.run("regress yy winv x1");
    assert_close(b, s.scalar("_b[winv]"), 1e-9);
    // fp generate
    s.run("fp generate double wq = w^(-1 -1)");
    s.run("gen double chk = wq_2 - ln(w)/w");
    s.run("summarize chk");
    assert_close(s.scalar("r(max)"), 0.0, 1e-12);
    // mfp keeps the clearly nonlinear term transformed
    let out = s.run("mfp: regress yy w x1");
    assert!(out.contains("Final multivariable fractional polynomial model"), "{out}");
}

#[test]
fn ztest_and_ratio_closed_forms() {
    let mut s = Session::new();
    // two-sample summary statistics: z = (20 - 15)/sqrt(25/20 + 16/32)
    let out = s.run("ztesti 20 20 5 32 15 4");
    assert!(out.contains("Two-sample z test"), "{out}");
    assert_close(s.scalar("r(z)"), 5.0 / (25.0f64 / 20.0 + 16.0 / 32.0).sqrt(), 1e-12);
    s.run("ztesti 24 62.6 15.8 75");
    assert_close(s.scalar("r(z)"), (62.6 - 75.0) / (15.8 / 24f64.sqrt()), 1e-12);
    // paired data (synthetic): checks against the closed forms
    let m1 = [14.0, 17.0, 16.0, 21.0, 13.0, 19.0, 15.0, 22.0, 18.0, 16.0];
    let m2 = [16.0, 18.0, 15.0, 24.0, 15.0, 22.0, 15.0, 23.0, 21.0, 17.0];
    let mut src = String::from("clear\ninput mpg1 mpg2\n");
    for i in 0..m1.len() {
        src.push_str(&format!("{} {}\n", m1[i], m2[i]));
    }
    src.push_str("end");
    s.script(&src);
    let n = m1.len() as f64;
    let (s1, s2): (f64, f64) = (m1.iter().sum(), m2.iter().sum());
    let dbar = (s1 - s2) / n;
    let out = s.run("ztest mpg1==mpg2, sd(2) corr(0.4)");
    assert!(out.contains("Paired z test"), "{out}");
    assert_close(s.scalar("r(z)"), dbar / ((4.0 + 4.0 - 2.0 * 0.4 * 4.0) / n).sqrt(), 1e-10);
    s.run("ztest mpg1==mpg2, unpaired sd(3)");
    assert_close(s.scalar("r(z)"), dbar / (3.0 * (2.0 / n).sqrt()), 1e-10);
    // ratio estimation on the same data: point estimate is the ratio of totals
    let out = s.run("ratio myratio: mpg1/mpg2");
    assert!(out.contains("Ratio estimation") && out.contains("myratio: mpg1/mpg2"), "{out}");
    let r = s1 / s2;
    assert_close(s.scalar("_b[myratio]"), r, 1e-10);
    // linearised SE: sqrt(sum((y1 - r y2)^2) / (n (n-1))) / mean(y2)
    let ss: f64 = (0..m1.len()).map(|i| (m1[i] - r * m2[i]).powi(2)).sum();
    let se = (ss / (n * (n - 1.0))).sqrt() / (s2 / n);
    assert_close(s.scalar("_se[myratio]"), se, 1e-8);
    s.run("test myratio = 1");
    assert_close(s.scalar("r(F)"), ((r - 1.0) / se).powi(2), 1e-6);
}

#[test]
fn symmetry_bowker_closed_form() {
    let mut s = Session::new();
    // synthetic 3x3 table; Bowker chi2 = sum_{i<j} (n_ij - n_ji)^2 / (n_ij + n_ji)
    let t: [[f64; 3]; 3] = [[30.0, 12.0, 20.0], [22.0, 40.0, 9.0], [11.0, 25.0, 18.0]];
    let out = s.run("symmi 30 12 20 \\ 22 40 9 \\ 11 25 18, contrib exact mh");
    assert!(out.contains("Symmetry (asymptotic)"), "{out}");
    let mut chi2 = 0.0;
    for i in 0..3 {
        for j in (i + 1)..3 {
            chi2 += (t[i][j] - t[j][i]).powi(2) / (t[i][j] + t[j][i]);
        }
    }
    assert_close(s.scalar("r(chi2)"), chi2, 1e-10);
    assert_close(s.scalar("r(df)"), 3.0, 0.0);
    assert!(s.scalar("r(p_exact)") > 0.0 && s.scalar("r(p_exact)") < 1.0);
}

#[test]
fn stem_and_leaf_layout() {
    let mut s = Session::new();
    s.run("set obs 12");
    s.run("gen x = 10 + mod(_n*7, 23)");
    let out = s.run("stem x, lines(2)");
    assert!(out.contains("Stem-and-leaf plot for x"), "{out}");
    // values 17 24 31 15 22 29 13 20 27 11 18 25
    assert!(out.contains("  1*   13"), "{out}");
    assert!(out.contains("  1.   578"), "{out}");
    assert!(out.contains("  2*   024"), "{out}");
    assert!(out.contains("  2.   579"), "{out}");
    assert!(out.contains("  3*   1"), "{out}");
}

#[test]
fn vwls_closed_form() {
    let mut s = Session::new();
    // synthetic data; vwls is WLS with weights 1/sd^2 and known variances
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let y = [2.1, 2.8, 4.4, 4.9, 6.3, 6.8, 8.5];
    let sd = [0.4, 0.4, 0.8, 0.8, 1.5, 1.5, 1.5];
    let mut src = String::from("input x y sd\n");
    for i in 0..x.len() {
        src.push_str(&format!("{} {} {}\n", x[i], y[i], sd[i]));
    }
    src.push_str("end");
    s.script(&src);
    let w: Vec<f64> = sd.iter().map(|v| 1.0 / (v * v)).collect();
    let sw: f64 = w.iter().sum();
    let xb = (0..7).map(|i| w[i] * x[i]).sum::<f64>() / sw;
    let yb = (0..7).map(|i| w[i] * y[i]).sum::<f64>() / sw;
    let sxx: f64 = (0..7).map(|i| w[i] * (x[i] - xb).powi(2)).sum();
    let sxy: f64 = (0..7).map(|i| w[i] * (x[i] - xb) * (y[i] - yb)).sum();
    let b = sxy / sxx;
    let a = yb - b * xb;
    let gof: f64 = (0..7).map(|i| w[i] * (y[i] - a - b * x[i]).powi(2)).sum();
    let out = s.run("vwls y x, sd(sd)");
    assert!(out.contains("Variance-weighted least-squares regression"), "{out}");
    assert_close(s.scalar("_b[x]"), b, 1e-6);
    assert_close(s.scalar("_se[x]"), (1.0 / sxx).sqrt(), 1e-6);
    assert_close(s.scalar("_b[_cons]"), a, 1e-6);
    assert_close(s.scalar("e(chi2_gf)"), gof, 1e-6);
    assert_close(s.scalar("e(chi2)"), b * b * sxx, 1e-5);
}

#[test]
fn ameans_closed_form() {
    let mut s = Session::new();
    s.script(
        "input x
         2
         8
         -3
         0
         .
         4
         end",
    );
    s.run("ameans x");
    assert_close(s.scalar("r(mean)"), 11.0 / 5.0, 1e-12);
    assert_close(s.scalar("r(N_pos)"), 3.0, 0.0);
    assert_close(s.scalar("r(mean_g)"), 64f64.powf(1.0 / 3.0), 1e-12);
    assert_close(s.scalar("r(mean_h)"), 3.0 / (1.0 / 2.0 + 1.0 / 8.0 + 1.0 / 4.0), 1e-12);
}

#[test]
fn tetrachoric_closed_form_and_edwards() {
    let mut s = Session::new();
    // balanced margins: Phi2(0,0,rho) = 1/4 + asin(rho)/(2 pi)
    s.script(
        "input a b w
         0 0 35
         0 1 15
         1 0 15
         1 1 35
         end",
    );
    s.run("expand w");
    let out = s.run("tetrachoric a b");
    assert!(out.contains("Tetrachoric rho"), "{out}");
    let rho = (2.0 * std::f64::consts::PI * (0.35 - 0.25)).sin();
    assert_close(s.scalar("r(rho)"), rho, 1e-6);
    assert_close(s.scalar("r(N)"), 100.0, 0.0);
    s.run("tetrachoric a b, edwards");
    let alpha = (35.0f64 * 35.0 / (15.0 * 15.0)).powf(std::f64::consts::PI / 4.0);
    assert_close(s.scalar("r(rho)"), (alpha - 1.0) / (alpha + 1.0), 1e-10);
}

#[test]
fn dydx_and_integ_published_example() {
    let mut s = Session::new();
    s.run("range x 0 12.56 100");
    s.run("gen y = exp(-x/6)*sin(x)");
    let out = s.run("integ y x, gen(iy)");
    assert!(out.contains("number of points = 100"), "{out}");
    // published value .85316397 (single-precision data); double data differ in the 8th digit
    assert_close(s.scalar("r(integral)"), 0.85316397, 2e-8);
    s.run("dydx y x, gen(dy)");
    s.run("gen double err = abs(dy - exp(-x/6)*(cos(x) - sin(x)/6))");
    s.run("summarize err in 10/90");
    assert!(s.scalar("r(max)") < 1e-3);
}

#[test]
fn fvset_changes_the_base_level() {
    let mut s = Session::new();
    data(&mut s, 90);
    s.run("regress y ib2.g");
    let b = s.scalar("_b[3.g]");
    s.run("fvset base 2 g");
    s.run("regress y i.g");
    assert_close(s.scalar("_b[3.g]"), b, 1e-10);
    let out = s.run("fvset report");
    assert!(out.contains('2'), "{out}");
    s.run("fvset clear g");
    s.run("regress y i.g");
    assert_close(s.scalar("_b[1b.g]"), 0.0, 0.0);
}

#[test]
fn fvrevar_creates_indicators() {
    let mut s = Session::new();
    data(&mut s, 30);
    s.run("fvrevar i.g x1");
    s.run("local v `r(varlist)'");
    let out = s.run("display \"`v'\"");
    let names: Vec<&str> = out.split_whitespace().collect();
    assert_eq!(names.len(), 4, "{out}");
    assert_eq!(names[3], "x1");
    s.run(&format!("gen byte chk = {} - (g == 2)", names[1]));
    s.run("summarize chk");
    assert_close(s.scalar("r(max)"), 0.0, 0.0);
    s.run("fvrevar i.g x1, list");
    let out = s.run("display \"`r(varlist)'\"");
    assert!(out.contains("g x1"), "{out}");
}

#[test]
fn smooth_rwgen_and_info_commands() {
    let mut s = Session::new();
    data(&mut s, 40);
    s.run("smooth 4253eh,twice y, gen(sy)");
    assert!(s.has_var("sy"));
    s.run("smooth 3 y, gen(s3)");
    // running median of span 3 at observation 5
    let y = s.nums("y");
    let mut w = [y[3], y[4], y[5]];
    w.sort_by(f64::total_cmp);
    assert_close(s.nums("s3")[4], w[1], 1e-12);
    s.run("rwgen jackknife jk, cluster(g)");
    assert!(s.has_var("jk1") && s.has_var("jk3") && !s.has_var("jk4"));
    s.run("summarize jk1 if g == 1");
    assert_close(s.scalar("r(max)"), 0.0, 0.0);
    s.run("rwgen bsample bw, reps(5) rseed(3)");
    s.run("summarize bw1");
    assert_close(s.scalar("r(sum)"), 40.0, 0.0);
    let out = s.run("query");
    assert!(out.contains("rngstream"), "{out}");
    s.run("about");
    s.run("more");
}

#[test]
fn reri_bayesboot_and_wildbootstrap_run() {
    let mut s = Session::new();
    data(&mut s, 400);
    s.run("gen byte a = mod(_n, 2)");
    s.run("gen byte c = mod(floor(_n/2), 2)");
    s.run("gen byte out = (u < 0.15 + 0.1*a + 0.1*c + 0.15*a*c)");
    let out = s.run("reri logit out a c");
    assert!(out.contains("RERI") && out.contains("Synergy ind."), "{out}");
    // RERI from the odds ratios of the fitted model
    s.run("logit out i.a##i.c");
    let (b1, b2, b3) = (s.scalar("_b[1.a]"), s.scalar("_b[1.c]"), s.scalar("_b[1.a#1.c]"));
    s.run("reri logit out a c");
    assert_close(s.scalar("r(reri)"), (b1 + b2 + b3).exp() - b1.exp() - b2.exp() + 1.0, 1e-8);
    let out = s.run("bayesboot, reps(20) rseed(1): regress y x1");
    assert!(out.contains("Bayesian bootstrap"), "{out}");
    assert!(s.scalar("_se[x1]") > 0.0);
    let out = s.run("wildbootstrap regress y x1, cluster(g) rseed(5) reps(99)");
    assert!(out.contains("Wild cluster bootstrap"), "{out}");
}

#[test]
fn ivqregress_with_own_instrument_is_qreg() {
    let mut s = Session::new();
    data(&mut s, 300);
    s.run("gen double z = x2 + u");
    s.run("gen double dd = z");
    s.run("qreg y z x1");
    let (bz, bx) = (s.scalar("_b[z]"), s.scalar("_b[x1]"));
    let out = s.run("ivqregress iqr y (z = dd) x1");
    assert!(out.contains("IV median regression") && out.contains("Inverse quantile regression"), "{out}");
    assert_close(s.scalar("_b[z]"), bz, 1e-5);
    assert_close(s.scalar("_b[x1]"), bx, 1e-5);
    let out = s.run("ivqregress smooth y (z = x2) x1");
    assert!(out.contains("Smoothed estimating equations"), "{out}");
    assert!(s.scalar("_se[z]") > 0.0);
}

#[test]
fn regress_time_series_estat_tests() {
    let mut s = Session::new();
    s.script(
        "set obs 40
         gen t = _n
         tsset t
         gen double x = mod(_n*7, 11)
         gen double u = invnormal((mod(_n*53, 97) + 0.5)/97)
         gen double y = 1 + x + u + 0.5*L.u
         regress y x
         predict double r, resid
         gen double d2 = (r - L.r)^2
         gen double r2 = r^2",
    );
    s.run("summarize d2");
    let num = s.scalar("r(sum)");
    s.run("summarize r2");
    let den = s.scalar("r(sum)");
    s.run("regress y x");
    let out = s.run("estat dwatson");
    assert!(out.contains("Durbin--Watson d-statistic(  2,    39)"), "{out}");
    assert_close(s.scalar("r(dw)"), num / den, 1e-10);
    // ARCH(1) LM = N R2 of e^2 on its lag
    s.run("gen double e2 = r^2");
    s.run("regress e2 L.e2");
    let lm = s.scalar("e(N)") * s.scalar("e(r2)");
    s.run("regress y x");
    s.run("estat archlm");
    assert_close(s.scalar("el(r(chi2),1,1)"), lm, 1e-8);
    let out = s.run("estat bgodfrey, small lags(1/2)");
    assert!(out.contains("Breusch--Godfrey LM test for autocorrelation"), "{out}");
    let out = s.run("estat durbinalt");
    assert!(out.contains("Durbin's alternative test"), "{out}");
}
