//! Nonlinear and dynamic panel models: xtlogit/xtprobit/xtpoisson/xtnbreg,
//! xtgls/xtpcse, xtabond/xtdpdsys/xtabond2 and the panel diagnostics.
//!
//! Conditional-logit and GEE reference values were computed independently
//! with statsmodels (ConditionalLogit; GEE with exchangeable correlation).

use ferrostat_tests::*;

fn binary_panel(s: &mut Session) {
    s.script(
        "set obs 600
         gen g = ceil(_n/10)
         gen t = _n - (g-1)*10
         gen double x = mod(_n*13, 17)/4 - 2
         gen double z = mod(_n*7, 5)/2
         gen double v = (mod(_n*29, 97) + 0.5)/97
         gen y = ((mod(g*7, 11) - 5)/2.5 + 0.8*x - 0.4*z + ln(v/(1-v))) > 0
         xtset g t",
    );
}

#[test]
fn xtlogit_fe_matches_conditional_logit() {
    let mut s = Session::new();
    binary_panel(&mut s);
    let out = s.run("xtlogit y x z, fe");
    assert!(out.contains("Conditional fixed-effects logistic regression"), "{out}");
    assert_close(s.scalar("e(ll)"), -208.42167222287816, 1e-8);
    assert_close(s.scalar("_b[x]"), 0.83333177, 1e-6);
    assert_close(s.scalar("_b[z]"), -0.55533896, 1e-6);
    assert_close(s.scalar("_se[x]"), 0.09387425, 1e-5);
    assert_close(s.scalar("_se[z]"), 0.14497695, 1e-5);
}

#[test]
fn xtlogit_pa_matches_gee_exchangeable() {
    // statsmodels divides the scale and the correlation by N - p; xtgee
    // divides by N, so agreement is to about 1e-4 at this sample size.
    let mut s = Session::new();
    binary_panel(&mut s);
    s.run("xtlogit y x z, pa");
    assert_close(s.scalar("_b[x]"), 0.66591321, 2e-4);
    assert_close(s.scalar("_b[z]"), -0.42326457, 2e-4);
    assert_close(s.scalar("_b[_cons]"), 0.08125994, 2e-4);
    assert_close(s.scalar("_se[x]"), 0.07176295, 2e-4);
    assert_close(s.scalar("_se[_cons]"), 0.1796695, 2e-4);
    s.run("xtlogit y x z, pa vce(robust)");
    assert_close(s.scalar("_se[x]"), 0.06961871, 2e-4);
    assert_close(s.scalar("_se[z]"), 0.11164561, 2e-4);
}

#[test]
fn xtlogit_re_quadrature_and_lr_test() {
    let mut s = Session::new();
    binary_panel(&mut s);
    let out = s.run("xtlogit y x z");
    assert!(out.contains("Random-effects logistic regression"), "{out}");
    assert!(out.contains("LR test of rho=0: chibar2(01) ="), "{out}");
    let ll = s.scalar("e(ll)");
    let llc = s.scalar("e(ll_c)");
    assert!(ll >= llc - 1e-6);
    assert_close(s.scalar("e(chi2_c)"), 2.0 * (ll - llc), 1e-6);
    // more quadrature points barely move the optimum
    s.run("xtlogit y x z, intpoints(30)");
    assert_close(s.scalar("e(ll)"), ll, 1e-4);
    let out = s.run("xtprobit y x z");
    assert!(out.contains("Random-effects probit regression"), "{out}");
    let r = s.scalar("e(rho)");
    assert!(r > 0.0 && r < 1.0);
}

fn count_panel(s: &mut Session) {
    s.script(
        "set obs 300
         gen g = ceil(_n/6)
         gen t = _n - (g-1)*6
         gen x = mod(_n*13, 17)/8 - 1
         gen u = mod(g*7, 11)/10 - 0.5
         gen y = floor(exp(0.5 + 0.6*x + u) * (0.4 + mod(_n*53, 97)/97))
         xtset g t",
    );
}

#[test]
fn xtpoisson_fe_equals_dummy_poisson() {
    let mut s = Session::new();
    count_panel(&mut s);
    s.run("xtpoisson y x, fe");
    let b = s.scalar("_b[x]");
    let se = s.scalar("_se[x]");
    s.run("poisson y x i.g");
    assert_close(b, s.scalar("_b[x]"), 1e-6);
    assert_close(se, s.scalar("_se[x]"), 1e-4);
}

#[test]
fn xtpoisson_xtnbreg_variants_run() {
    let mut s = Session::new();
    count_panel(&mut s);
    for c in ["xtpoisson y x", "xtpoisson y x, normal", "xtpoisson y x, pa", "xtnbreg y x", "xtnbreg y x, fe"] {
        let out = s.run(c);
        assert!(out.contains("Number of groups ="), "{c}: {out}");
        assert_eq!(s.scalar("e(N_g)"), 50.0, "{c}");
    }
    s.run("xtpoisson y x, irr");
    s.run("predict xb, xb");
    assert!(s.has_var("xb"));
}

fn dynamic_panel(s: &mut Session) {
    s.script(
        "set obs 400
         gen id = ceil(_n/8)
         gen year = 2000 + _n - (id-1)*8
         gen x = mod(_n*13, 17)/4 - 2 + mod(id, 5)/3
         gen e = (mod(_n*29, 97) - 48)/60
         gen a = (mod(id*7, 11) - 5)/5
         gen y = a + x + e
         xtset id year
         replace y = 0.5*L.y + 0.4*x + a + e if year > 2001",
    );
}

#[test]
fn xtabond_equals_xtabond2_difference_gmm() {
    let mut s = Session::new();
    dynamic_panel(&mut s);
    let out = s.run("xtabond y x, vce(robust) noconstant");
    assert!(out.contains("Arellano\u{2013}Bond dynamic panel-data estimation"), "{out}");
    let b = s.scalar("_b[L.y]");
    let se = s.scalar("_se[L.y]");
    let bx = s.scalar("_b[x]");
    let ar1 = s.scalar("e(ar1)");
    s.run("xtabond2 y L.y x, gmm(L.y, laglimits(1 .)) iv(x) nolevel robust noconstant");
    assert_close(s.scalar("_b[L.y]"), b, 1e-8);
    assert_close(s.scalar("_se[L.y]"), se, 1e-6);
    assert_close(s.scalar("_b[x]"), bx, 1e-8);
    assert_close(s.scalar("e(ar1)"), ar1, 1e-6);
    let out = s.run("estat abond");
    assert!(out.contains("H0: No autocorrelation"), "{out}");
}

#[test]
fn xtabond_twostep_and_sargan() {
    let mut s = Session::new();
    dynamic_panel(&mut s);
    let out = s.run("xtabond y x, twostep vce(robust)");
    assert!(out.contains("WC-Robust"), "{out}");
    s.run("xtabond y x");
    let out = s.run("estat sargan");
    assert!(out.contains("Sargan test of overidentifying restrictions"), "{out}");
    let df = s.scalar("r(df)");
    assert_eq!(df, s.scalar("e(zrank)") - 3.0);
    let out = s.run("xtdpdsys y x");
    assert!(out.contains("System dynamic panel-data estimation"), "{out}");
}

#[test]
fn xtgls_iid_is_ols_and_pcse_keeps_ols_coefficients() {
    let mut s = Session::new();
    dynamic_panel(&mut s);
    s.run("regress y x");
    let b = s.scalar("_b[x]");
    let rss = s.scalar("e(rss)");
    let n = s.scalar("e(N)");
    s.run("xtgls y x");
    assert_close(s.scalar("_b[x]"), b, 1e-9);
    s.run("xtpcse y x");
    assert_close(s.scalar("_b[x]"), b, 1e-9);
    s.run("xtpcse y x, independent");
    // independent PCSE = OLS variance with sigma^2 = rss/N
    s.run("scalar se_pcse = _se[x]");
    let se_pcse = s.scalar("scalar(se_pcse)");
    s.run("regress y x");
    let se_ols = s.scalar("_se[x]") * ((n - 2.0) / n).sqrt();
    assert_close(se_pcse, se_ols, 1e-8);
    let _ = rss;
    s.run("xtgls y x, panels(hetero) corr(psar1)");
    assert_eq!(s.scalar("e(N_g)"), 50.0);
}

#[test]
fn panel_diagnostics_run() {
    let mut s = Session::new();
    dynamic_panel(&mut s);
    s.run("xtreg y x, re");
    let out = s.run("xttest0");
    assert!(out.contains("chibar2(01)"), "{out}");
    s.run("xtreg y x, fe");
    let out = s.run("xttest3");
    assert!(out.contains("Modified Wald test"), "{out}");
    assert_eq!(s.scalar("r(df)"), 50.0);
    let out = s.run("xtcsd, pesaran abs");
    assert!(out.contains("Pesaran's test"), "{out}");
    let out = s.run("xtserial y x");
    assert!(out.contains("Wooldridge test"), "{out}");
    assert_eq!(s.scalar("r(df_r)"), 49.0);
}
