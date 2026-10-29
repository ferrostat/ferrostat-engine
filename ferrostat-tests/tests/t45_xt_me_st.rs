//! Panel GEE / AR(1) / Hausman-Taylor / random-coefficient models, panel
//! description and unit-root tests, random-intercept ordinal, censored,
//! count and survival models (xt and me), and further survival commands.

use ferrostat_tests::*;

fn lin_panel(s: &mut Session) {
    s.script(
        "set obs 400
         gen g = ceil(_n/10)
         gen t = _n - (g-1)*10
         gen double x = mod(_n*13, 17)/4 - 2
         gen double z = mod(_n*7, 5)/2
         gen double w = mod(g*3, 7)
         gen double e = (mod(_n*29, 97) - 48)/30
         gen double u = (mod(g*7, 11) - 5)/3
         gen double y = 1 + 0.8*x - 0.5*z + 0.3*w + u + e
         xtset g t",
    );
}

#[test]
fn xtgee_independent_gaussian_equals_regress() {
    let mut s = Session::new();
    lin_panel(&mut s);
    s.run("regress y x z");
    let (b, se, c) = (s.scalar("_b[x]"), s.scalar("_se[x]"), s.scalar("_b[_cons]"));
    let out = s.run("xtgee y x z, corr(independent) nmp");
    assert!(out.contains("GEE population-averaged model"), "{out}");
    assert_close(s.scalar("_b[x]"), b, 1e-8);
    assert_close(s.scalar("_b[_cons]"), c, 1e-8);
    assert_close(s.scalar("_se[x]"), se, 1e-6);
    // default scale divides by N
    s.run("xtgee y x z, corr(independent)");
    assert_close(s.scalar("_se[x]"), se * (397.0f64 / 400.0).sqrt(), 1e-6);
}

#[test]
fn xtgee_correlations_and_families() {
    let mut s = Session::new();
    lin_panel(&mut s);
    for c in ["corr(exchangeable)", "corr(ar 1)", "corr(stationary 2)", "corr(unstructured)", "corr(exch) vce(robust)"] {
        let out = s.run(&format!("xtgee y x z, {c}"));
        assert!(out.contains("Wald chi2(2)"), "{c}: {out}");
        assert_eq!(s.scalar("e(N_g)"), 40.0);
    }
    s.run("xtgee y x z, corr(ar 1)");
    let out = s.run("estat wcorrelation");
    assert!(out.contains("r10"), "{out}");
    // exchangeable binomial GEE matches xtlogit, pa
    s.run("gen yb = y > 1.5");
    s.run("xtlogit yb x z, pa");
    let b = s.scalar("_b[x]");
    s.run("xtgee yb x z, family(binomial) link(logit) corr(exchangeable)");
    assert_close(s.scalar("_b[x]"), b, 1e-6);
    s.run("gen yc = floor(exp(0.2*x + 0.1*z))");
    s.run("xtgee yc x z, family(poisson) corr(exch)");
    assert!(s.scalar("e(phi)") == 1.0);
    s.run("xtcloglog yb x z, pa");
    assert_eq!(s.scalar("e(N)"), 400.0);
}

#[test]
fn xtregar_with_rho_zero_equals_within() {
    let mut s = Session::new();
    lin_panel(&mut s);
    s.run("xtregar y x z, fe rhof(0)");
    let (b, se) = (s.scalar("_b[x]"), s.scalar("_se[x]"));
    assert_eq!(s.scalar("e(N)"), 360.0);
    // the first observation of each panel is lost
    s.run("xtreg y x z if t > 1, fe");
    assert_close(b, s.scalar("_b[x]"), 1e-8);
    assert_close(se, s.scalar("_se[x]"), 1e-6);
    let out = s.run("xtregar y x z, re lbi");
    assert!(out.contains("RE GLS regression with AR(1) disturbances"), "{out}");
    assert!(out.contains("Baltagi-Wu LBI"), "{out}");
}

#[test]
fn xthtaylor_and_xtrc_run() {
    let mut s = Session::new();
    lin_panel(&mut s);
    s.run("gen double x2 = x + u*z");
    s.run("gen double w2 = w + u");
    let out = s.run("xthtaylor y x x2 z w w2, endog(x2 w2)");
    assert!(out.contains("Hausman-Taylor estimation"), "{out}");
    assert!(out.contains("TIendogenous"), "{out}");
    assert!(s.scalar("e(sigma_u)") > 0.0);
    let out = s.run("xtrc y x z");
    assert!(out.contains("Test of parameter constancy"), "{out}");
    assert_eq!(s.scalar("e(df_chi2c)"), 3.0 * 39.0);
}

#[test]
fn panel_description_commands() {
    let mut s = Session::new();
    lin_panel(&mut s);
    s.run("drop if t == 5 & g < 4");
    let out = s.run("xtdescribe");
    assert!(out.contains("Distribution of T_i"), "{out}");
    assert_eq!(s.scalar("r(n)"), 40.0);
    assert_eq!(s.scalar("r(min)"), 9.0);
    s.run("xtsum x");
    assert_eq!(s.scalar("r(n)"), 40.0);
    // within sd is the sd of x - xbar_i + xbar
    let sdw = s.scalar("r(sd_w)");
    s.script("bysort g: egen double mx = mean(x)\n summarize x\n gen double wx = x - mx + r(mean)\n summarize wx");
    assert_close(sdw, s.scalar("r(sd)"), 1e-10);
    s.run("gen byte d = x > 0");
    let out = s.run("xttab d");
    assert!(out.contains("(n = 40)"), "{out}");
    let out = s.run("xttrans d, freq");
    assert!(out.contains("Total"), "{out}");
    s.run("xtdata y x z, fe clear");
    s.run("xtreg y x z, fe");
    let b = s.scalar("_b[x]");
    s.run("regress y x z");
    assert_close(s.scalar("_b[x]"), b, 1e-8);
}

#[test]
fn unit_root_tests_run() {
    let mut s = Session::new();
    s.script(
        "set obs 300
         gen g = ceil(_n/30)
         gen t = _n - (g-1)*30
         gen double e = (mod(_n*29, 97) - 48)/30
         gen double y = e
         xtset g t
         replace y = 0.5*L.y + e if t > 1",
    );
    for (c, key) in [("ht", "z"), ("llc", "td"), ("breitung", "lambda"), ("ips", "zttildebar"), ("hadri", "z")] {
        s.run(&format!("xtunitroot {c} y"));
        assert!(s.scalar(&format!("r({key})")).is_finite(), "{c}");
    }
    // a stationary AR(0.5) panel: unit roots rejected
    s.run("xtunitroot ht y");
    assert!(s.scalar("r(p)") < 0.01);
    s.run("xtunitroot fisher y, dfuller lags(1)");
    assert!(s.scalar("r(p_Z)") < 0.05);
}

fn ord_panel(s: &mut Session) {
    s.script(
        "set obs 600
         gen g = ceil(_n/12)
         gen t = _n - (g-1)*12
         gen double x = mod(_n*13, 17)/4 - 2
         gen double v = (mod(_n*29, 97) + 0.5)/97
         gen double u = (mod(g*7, 11) - 5)/4
         gen double ys = 0.9*x + u + ln(v/(1-v))
         gen y = 1 + (ys > -1) + (ys > 0.5) + (ys > 2)
         xtset g t",
    );
}

#[test]
fn xtologit_and_meologit_agree_and_nest_ologit() {
    let mut s = Session::new();
    ord_panel(&mut s);
    s.run("ologit y x");
    let ll0 = s.scalar("e(ll)");
    let out = s.run("xtologit y x");
    assert!(out.contains("Random-effects ordered logistic regression"), "{out}");
    assert!(out.contains("LR test vs. ologit model"), "{out}");
    assert_close(s.scalar("e(ll_c)"), ll0, 1e-7);
    assert!(s.scalar("e(ll)") > ll0);
    assert_close(s.scalar("e(chi2_c)"), 2.0 * (s.scalar("e(ll)") - ll0), 1e-8);
    let ll = s.scalar("e(ll)");
    let b = s.scalar("_b[x]");
    s.run("meologit y x || g:, intpoints(12)");
    assert_close(s.scalar("e(ll)"), ll, 1e-6);
    assert_close(s.scalar("_b[x]"), b, 1e-5);
    let out = s.run("estat icc");
    assert!(out.contains("Intraclass correlation"), "{out}");
    assert!(s.scalar("r(icc)") > 0.0 && s.scalar("r(icc)") < 1.0);
    // more quadrature points barely move the fit
    s.run("xtologit y x, intpoints(30)");
    assert_close(s.scalar("e(ll)"), ll, 1e-4);
    let out = s.run("xtoprobit y x");
    assert!(out.contains("ordered probit"), "{out}");
    s.run("meoprobit y x || g:");
    assert!(s.scalar("e(k_cat)") == 4.0);
}

#[test]
fn censored_count_cloglog_random_effects() {
    let mut s = Session::new();
    lin_panel(&mut s);
    s.run("xttobit y x z, ul(3)");
    let ll = s.scalar("e(ll)");
    assert!(s.scalar("e(N_rc)") > 0.0);
    s.run("metobit y x z || g:, ul(3) intpoints(12)");
    assert_close(s.scalar("e(ll)"), ll, 1e-6);
    // interval regression with point data and right censoring at 3 is tobit
    s.run("gen double lo = cond(y >= 3, 3, y)");
    s.run("gen double hi = cond(y >= 3, ., y)");
    s.run("xtintreg lo hi x z");
    assert_close(s.scalar("e(ll)"), ll, 1e-6);
    s.run("meintreg lo hi x z || g:, intpoints(12)");
    assert_close(s.scalar("e(ll)"), ll, 1e-6);
    s.run("gen yb = y > 1.5");
    let out = s.run("xtcloglog yb x z");
    assert!(out.contains("Random-effects complementary log-log model"), "{out}");
    let ll = s.scalar("e(ll)");
    s.run("mecloglog yb x z || g:, intpoints(12)");
    assert_close(s.scalar("e(ll)"), ll, 1e-6);
    s.run("gen yc = floor(exp(0.3*x + u/2) * (0.5 + mod(_n*53, 97)/97))");
    let out = s.run("menbreg yc x || g:");
    assert!(out.contains("Mixed-effects negative binomial regression"), "{out}");
    assert!(s.scalar("e(k_r)") == 1.0);
}

fn surv_data(s: &mut Session) {
    s.script(
        "set obs 240
         gen id = _n
         gen g = ceil(_n/4)
         gen double x = mod(_n*13, 17)/8 - 1
         gen double v = (mod(_n*29, 97) + 0.5)/97
         gen double u = (mod(g*7, 11) - 5)/5
         gen double time = (-ln(v)/exp(0.5*x + u))^(1/1.3) * 10
         gen byte fail = mod(_n, 5) != 0
         gen byte cause = cond(fail, 1 + (mod(_n, 3) == 0), 0)",
    );
}

#[test]
fn stcrreg_without_competing_events_is_cox_breslow() {
    let mut s = Session::new();
    surv_data(&mut s);
    s.run("stset time, failure(fail)");
    s.run("gen byte none = 0");
    let out = s.run("stcrreg x, compete(none == 1)");
    assert!(out.contains("Competing-risks regression"), "{out}");
    let (b, ll) = (s.scalar("_b[x]"), s.scalar("e(ll)"));
    s.run("stcox x, breslow nohr");
    assert_close(b, s.scalar("_b[x]"), 1e-7);
    assert_close(ll, s.scalar("e(ll)"), 1e-7);
    // with competing events the subdistribution fit differs
    s.run("stset time, failure(cause == 1)");
    s.run("stcrreg x, compete(cause == 2)");
    assert!(s.scalar("e(N_compete)") > 0.0);
    assert!((s.scalar("_b[x]") - b).abs() > 1e-4);
}

#[test]
fn mestreg_xtstreg_and_stintreg() {
    let mut s = Session::new();
    surv_data(&mut s);
    s.run("stset time, failure(fail)");
    s.run("streg x, distribution(weibull) nohr");
    let (b, ll) = (s.scalar("_b[x]"), s.scalar("e(ll)"));
    // exact times as zero-width intervals reproduce streg
    s.run("gen double tl = cond(fail, time, time)");
    s.run("gen double tu = cond(fail, time, .)");
    s.run("stintreg x, interval(tl tu) distribution(weibull)");
    assert_close(s.scalar("_b[x]"), b, 1e-5);
    assert_close(s.scalar("e(ll)"), ll, 1e-6);
    s.run("mestreg x || g:, distribution(weibull) intpoints(12)");
    let ll2 = s.scalar("e(ll)");
    assert!(ll2 >= ll - 1e-6);
    s.run("xtset g");
    s.run("xtstreg x, distribution(weibull)");
    assert_close(s.scalar("e(ll)"), ll2, 1e-6);
    s.run("mestreg x || g:, distribution(exponential)");
    assert!(s.scalar("e(converged)") == 1.0);
}

#[test]
fn st_utilities() {
    let mut s = Session::new();
    surv_data(&mut s);
    s.run("stset time, failure(fail) id(id)");
    s.run("stgen ever_hi = ever(x > 0.5)");
    assert!(s.has_var("ever_hi"));
    let out = s.run("stvary x");
    assert!(out.contains("constant"), "{out}");
    assert_eq!(s.scalar("r(cons)"), 240.0);
    s.run("gen byte expo = x > 0");
    let out = s.run("stir expo");
    assert!(out.contains("Inc. rate ratio"), "{out}");
    let irr = s.scalar("r(irr)");
    s.run("stmh expo");
    assert_close(s.scalar("r(RR)"), irr, 1e-10);
    // count-time data
    let mut c = Session::new();
    c.script(
        "set obs 3
         gen t = _n
         gen nf = 2*_n
         gen nc = 1
         ctset t nf nc
         cttost",
    );
    assert_eq!(c.n(), 6);
    assert_eq!(c.scalar("_N"), 6.0);
}

#[test]
fn xtfrontier_xtline_quadchk() {
    let mut s = Session::new();
    lin_panel(&mut s);
    s.run("gen double yf = 1 + 0.5*x + e/3 - abs(u)/2");
    let out = s.run("xtfrontier yf x, ti");
    assert!(out.contains("Time-invariant inefficiency model"), "{out}");
    let ll = s.scalar("e(ll)");
    s.run("xtfrontier yf x, tvd");
    assert!(s.scalar("e(ll)") >= ll - 1e-6);
    s.run("xtline y if g < 3");
    s.run("xtline y if g < 3, overlay");
    s.run("gen yb = y > 1.5");
    s.run("xtlogit yb x z");
    let out = s.run("quadchk");
    assert!(out.contains("Quadrature check"), "{out}");
}
