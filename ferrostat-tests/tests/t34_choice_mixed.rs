//! Selection models (heckman, heckprobit), bivariate probit, conditional
//! logit (clogit, asclogit), linear mixed models (mixed) and the mixed GLM
//! family (meprobit, mepoisson, meglm).
//!
//! Reference values were computed independently: selection and bivariate
//! probit likelihoods maximised with scipy (bivariate normal from
//! scipy.stats.multivariate_normal), clogit with statsmodels
//! ConditionalLogit, mixed models with statsmodels MixedLM (ML and REML,
//! random intercepts / unstructured random slopes, BLUPs).

use ferrostat_tests::*;

fn selection_data(s: &mut Session) {
    s.script(
        "set obs 500
         gen double x1 = mod(_n*13, 17)/4 - 2
         gen double z1 = mod(_n*7, 11)/5 - 1
         gen double e1 = invnormal((mod(_n*37, 101)+0.5)/101)
         gen double e2 = 0.6*e1 + 0.8*invnormal((mod(_n*53, 103)+0.5)/103)
         gen s = (0.3 + 0.8*z1 + 0.4*x1 + e2) > 0
         gen double y = 1 + 0.5*x1 + 1.5*e1 if s
         gen yb = (0.2 + 0.7*x1 + e1) > 0 if s
         gen y1 = (0.2 + 0.7*x1 + e1) > 0",
    );
}

#[test]
fn heckman_ml_matches_direct_maximisation() {
    let mut s = Session::new();
    selection_data(&mut s);
    let out = s.run("heckman y x1, select(s = z1 x1)");
    assert!(out.contains("Heckman selection model"), "{out}");
    assert!(out.contains("LR test of indep. eqns. (rho = 0):"), "{out}");
    assert_close(s.scalar("e(ll)"), -784.6772231256066, 1e-8);
    assert_close(s.scalar("_b[x1]"), 0.5426725237, 1e-6);
    assert_close(s.scalar("_b[_cons]"), 1.0063847419, 1e-6);
    assert_close(s.scalar("_b[select:z1]"), 0.9428254539, 1e-6);
    assert_close(s.scalar("_b[select:_cons]"), 0.3294629894, 1e-6);
    assert_close(s.scalar("e(rho)"), 0.7050035245f64.tanh(), 1e-6);
    assert_close(s.scalar("e(sigma)"), 0.3898116708f64.exp(), 1e-6);
    assert_close(s.scalar("_se[x1]"), 0.0743370387, 1e-4);
    assert_close(s.scalar("_se[select:z1]"), 0.1027422365, 1e-4);
    assert_eq!(s.scalar("e(N_selected)"), 303.0);
    assert!(s.scalar("e(chi2_c)") > 0.0);
}

#[test]
fn heckman_twostep_and_mills() {
    let mut s = Session::new();
    selection_data(&mut s);
    let out = s.run("heckman y x1, select(s = z1 x1) twostep mills(mr)");
    assert!(out.contains("two-step estimates"), "{out}");
    assert_close(s.scalar("_b[x1]"), 0.6054669162, 1e-7);
    assert_close(s.scalar("_b[_cons]"), 0.7917229703, 1e-7);
    assert_close(s.scalar("e(lambda)"), 1.2782892631, 1e-7);
    assert_close(s.scalar("e(sigma)"), 1.5840493525747252, 1e-7);
    assert_close(s.scalar("e(rho)"), 0.80697565453663, 1e-6);
    assert!(s.has_var("mr"));
    s.run("predict yc, ycond");
    assert!(s.has_var("yc"));
}

#[test]
fn heckprobit_matches_direct_maximisation() {
    let mut s = Session::new();
    selection_data(&mut s);
    let out = s.run("heckprobit yb x1, select(s = z1 x1)");
    assert!(out.contains("Probit model with sample selection"), "{out}");
    assert_close(s.scalar("e(ll)"), -391.5789937455058, 1e-8);
    assert_close(s.scalar("_b[x1]"), 0.7749716051, 1e-6);
    assert_close(s.scalar("_b[select:x1]"), 0.4035175818, 1e-6);
    assert_close(s.scalar("e(rho)"), 0.8208781376f64.tanh(), 1e-6);
    assert_close(s.scalar("_se[x1]"), 0.0783195559, 1e-4);
}

#[test]
fn biprobit_both_syntaxes() {
    let mut s = Session::new();
    selection_data(&mut s);
    let out = s.run("biprobit (y1 = x1) (s = z1 x1)");
    assert!(out.contains("Seemingly unrelated bivariate probit"), "{out}");
    assert!(out.contains("LR test of rho=0"), "{out}");
    assert_close(s.scalar("e(ll)"), -493.42673183275286, 1e-8);
    assert_close(s.scalar("_b[x1]"), 0.7106264638, 1e-6);
    assert_close(s.scalar("_b[s:z1]"), 0.9239432046, 1e-6);
    assert_close(s.scalar("e(rho)"), 0.6935991528f64.tanh(), 1e-6);
    assert_close(s.scalar("_se[x1]"), 0.0588410377, 1e-4);
    s.run("predict p11");
    s.run("predict p00, p00");
    // same regressors: the short syntax fits the same kind of model
    let out = s.run("biprobit y1 s x1");
    assert!(out.contains("Bivariate probit regression"), "{out}");
    assert_eq!(s.scalar("e(N)"), 500.0);
}

#[test]
fn clogit_multiple_positives_matches_statsmodels() {
    let mut s = Session::new();
    s.script(
        "set obs 300
         gen g = ceil(_n/5)
         gen double x = mod(_n*13, 17)/4 - 2
         gen double z = mod(_n*7, 5)/2
         gen double v = (mod(_n*29, 97)+0.5)/97
         gen y = (0.8*x - 0.4*z + ln(v/(1-v))) > 0.5",
    );
    let out = s.run("clogit y x z, group(g)");
    assert!(out.contains("multiple positive outcomes"), "{out}");
    assert!(out.contains("4 groups (20 obs) dropped"), "{out}");
    assert_close(s.scalar("e(ll)"), -82.63573019373995, 1e-9);
    assert_close(s.scalar("_b[x]"), 0.78634588, 1e-6);
    assert_close(s.scalar("_b[z]"), -0.45673402, 1e-6);
    assert_close(s.scalar("_se[x]"), 0.12804828, 1e-6);
    assert_close(s.scalar("_se[z]"), 0.1952674, 1e-6);
    assert_eq!(s.scalar("e(N)"), 280.0);
    let out = s.run("clogit y x z, group(g) or vce(robust)");
    assert!(out.contains("Odds ratio"), "{out}");
    assert!(out.contains("clusters in g"), "{out}");
    s.run("predict p, pc1");
    // pc1 sums to one within each group
    s.run("egen sp = total(p), by(g)");
    let sp = s.nums("sp");
    assert!(sp.iter().all(|v| (v - 1.0).abs() < 1e-9));
}

#[test]
fn asclogit_equals_clogit_with_alternative_dummies() {
    let mut s = Session::new();
    s.script(
        "set obs 240
         gen id = ceil(_n/3)
         gen alt = _n - (id-1)*3
         gen double cost = mod(_n*7, 13)/3
         gen double inc = mod(id*11, 17)/4
         gen double v = (mod(_n*29, 97)+0.5)/97
         gen double u = -0.6*cost + (alt==2)*(0.3*inc - 1) + (alt==3)*(-0.2*inc + 0.5) - ln(-ln(v))
         egen double mu = max(u), by(id)
         gen ch = u == mu",
    );
    s.run("asclogit ch cost, case(id) alternatives(alt) casevars(inc) basealternative(1)");
    let ll = s.scalar("e(ll)");
    let bc = s.scalar("_b[cost]");
    s.script(
        "gen a2 = alt==2
         gen a3 = alt==3
         gen i2 = inc*a2
         gen i3 = inc*a3",
    );
    s.run("clogit ch cost i2 a2 i3 a3, group(id)");
    assert_close(s.scalar("e(ll)"), ll, 1e-9);
    assert_close(s.scalar("_b[cost]"), bc, 1e-7);
}

fn mixed_data(s: &mut Session) {
    s.script(
        "set obs 320
         gen id = ceil(_n/8)
         gen t = _n - (id-1)*8
         gen region = ceil(id/5)
         gen double x = mod(_n*13, 17)/4 - 2
         gen double u0 = invnormal((mod(id*37, 41)+0.5)/41)
         gen double u1 = 0.5*invnormal((mod(id*17, 43)+0.5)/43)
         gen double r0 = 0.7*invnormal((mod(region*5, 9)+0.5)/9)
         gen double y = 2 + 0.5*x + u0 + u1*t + r0 + invnormal((mod(_n*53, 331)+0.5)/331)",
    );
}

#[test]
fn mixed_random_intercept_ml_and_reml() {
    let mut s = Session::new();
    mixed_data(&mut s);
    let out = s.run("mixed y x || id:");
    assert!(out.contains("Mixed-effects ML regression"), "{out}");
    assert!(out.contains("id: Identity"), "{out}");
    assert!(out.contains("LR test vs. linear model: chibar2(01)"), "{out}");
    assert_close(s.scalar("e(ll)"), -653.9667587061692, 1e-8);
    assert_close(s.scalar("_b[x]"), 0.5051467379, 1e-6);
    assert_close(s.scalar("_b[_cons]"), 2.3583824567, 1e-6);
    assert_close(s.scalar("exp(2*_b[lns1_1_1:_cons])"), 4.9130863254, 1e-3);
    assert_close(s.scalar("exp(2*_b[lnsig_e:_cons])"), 2.4468412199566862, 1e-4);
    let out = s.run("estat icc");
    assert!(out.contains("Residual intraclass correlation"), "{out}");
    let icc = s.scalar("r(icc2)");
    assert!(icc > 0.6 && icc < 0.7, "{icc}");
    let out = s.run("mixed y x || id:, reml");
    assert!(out.contains("Log restricted-likelihood"), "{out}");
    assert_close(s.scalar("e(ll)"), -655.7640073499243, 1e-8);
    assert_close(s.scalar("_se[x]"), 0.072645216, 1e-4);
    assert_close(s.scalar("exp(2*_b[lns1_1_1:_cons])"), 5.0459989726, 1e-3);
}

#[test]
fn mixed_random_slope_unstructured_and_blups() {
    let mut s = Session::new();
    mixed_data(&mut s);
    let out = s.run("mixed y x || id: t, covariance(unstructured)");
    assert!(out.contains("id: Unstructured"), "{out}");
    assert!(out.contains("cov(t,_cons)"), "{out}");
    assert_close(s.scalar("e(ll)"), -572.450462957107, 1e-8);
    assert_close(s.scalar("_b[x]"), 0.4916256592, 1e-6);
    assert_close(s.scalar("_se[x]"), 0.0514109612, 1e-4);
    assert_close(s.scalar("exp(2*_b[lns1_1_1:_cons])"), 0.2181221038, 1e-4);
    assert_close(s.scalar("exp(2*_b[lns1_1_2:_cons])"), 1.5828225677, 1e-4);
    s.run("predict b_t b_c, reffects");
    let bt = s.nums("b_t");
    let bc = s.nums("b_c");
    // id 1 is observation 1, id 7 is observation 49
    assert_close(bt[0], -0.0017123522, 1e-4);
    assert_close(bc[0], 0.755355851, 1e-4);
    assert_close(bt[48], 0.5285187286, 1e-4);
    assert_close(bc[48], -1.782898592, 1e-4);
    s.run("predict fit, fitted");
    s.run("predict res, residuals");
    let (y, f, r) = (s.nums("y"), s.nums("fit"), s.nums("res"));
    assert!((0..y.len()).all(|i| (y[i] - f[i] - r[i]).abs() < 1e-9));
    let out = s.run("mixed y x || id: t, covariance(unstructured) reml");
    assert_close(s.scalar("e(ll)"), -575.052876564887, 1e-7);
    assert!(out.contains("REML"), "{out}");
}

#[test]
fn mixed_nested_levels() {
    let mut s = Session::new();
    mixed_data(&mut s);
    let out = s.run("mixed y x || region: || id:");
    assert!(out.contains("Group variable |     groups"), "{out}");
    assert!(out.contains("region: Identity"), "{out}");
    assert!(out.contains("Note: LR test is conservative"), "{out}");
    let ll = s.scalar("e(ll)");
    // the region level adds a parameter: never worse than the two-level fit
    s.run("estimates store three");
    s.run("mixed y x || id:");
    assert!(ll >= s.scalar("e(ll)") - 1e-8);
    s.run("mixed y x || region: || id:");
    let out = s.run("estat icc");
    assert!(out.contains("id|region"), "{out}");
}

#[test]
fn meprobit_mepoisson_match_xt_random_effects() {
    let mut s = Session::new();
    s.script(
        "set obs 600
         gen g = ceil(_n/10)
         gen t = _n - (g-1)*10
         gen double x = mod(_n*13, 17)/4 - 2
         gen double v = (mod(_n*29, 97) + 0.5)/97
         gen yb = ((mod(g*7, 11) - 5)/2.5 + 0.8*x + invnormal(v)) > 0
         gen yc = floor(exp(0.3 + 0.4*x + (mod(g*7, 11)-5)/10) * (0.3 + mod(_n*53, 97)/70))
         xtset g t",
    );
    s.run("xtprobit yb x, re intpoints(7)");
    let ll = s.scalar("e(ll)");
    let bx = s.scalar("_b[x]");
    let out = s.run("meprobit yb x || g:");
    assert!(out.contains("Mixed-effects probit regression"), "{out}");
    assert_close(s.scalar("e(ll)"), ll, 1e-7);
    assert_close(s.scalar("_b[x]"), bx, 1e-6);
    let out = s.run("meglm yb x || g:, family(bernoulli) link(probit)");
    assert!(out.contains("Mixed-effects GLM"), "{out}");
    assert_close(s.scalar("e(ll)"), ll, 1e-7);
    s.run("xtpoisson yc x, re normal intpoints(7)");
    let ll = s.scalar("e(ll)");
    let bx = s.scalar("_b[x]");
    let out = s.run("mepoisson yc x || g:");
    assert!(out.contains("Mixed-effects Poisson regression"), "{out}");
    assert_close(s.scalar("e(ll)"), ll, 1e-7);
    assert_close(s.scalar("_b[x]"), bx, 1e-6);
    s.fails("meglm yb x || g:, family(gaussian)");
}
