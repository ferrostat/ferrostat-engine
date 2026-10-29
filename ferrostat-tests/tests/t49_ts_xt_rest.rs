//! Remaining time-series, panel, survival and multilevel commands:
//! forecast models with simulation, varbasic, estat acplot, xtdpd,
//! panel cointegration tests, extended regression models (cross-section
//! and panel), panel multinomial logit, panel VAR, survival utilities and
//! nonlinear mixed-effects regression.

use ferrostat_tests::*;

fn ts_data(s: &mut Session, n: usize) {
    s.script(&format!(
        "set obs {n}
         set seed 3
         gen t = _n
         tsset t
         gen double x = rnormal()
         gen double y = 0
         replace y = 1 + 0.5*L.y + x + rnormal() in 2/l"
    ));
}

#[test]
fn forecast_coefvector_dynamic_solution_matches_recursion() {
    let mut s = Session::new();
    ts_data(&mut s, 40);
    s.script(
        "tsappend, add(3)
         replace x = 1 if missing(x)
         matrix c = (0.5, 2, 1)
         matrix colnames c = y:L.y y:x y:_cons
         forecast create m1
         forecast coefvector c
         forecast solve, begin(41)",
    );
    let y40 = s.scalar("y[40]");
    let f1 = 0.5 * y40 + 2.0 + 1.0;
    let f2 = 0.5 * f1 + 3.0;
    assert_close(s.scalar("f_y[41]"), f1, 1e-10);
    assert_close(s.scalar("f_y[42]"), f2, 1e-10);
    assert_close(s.scalar("f_y[40]"), y40, 1e-12);
    let out = s.run("forecast list");
    assert!(out.contains("forecast coefvector c"), "{out}");
    s.run("forecast drop");
    assert!(!s.has_var("f_y"));
    s.run("forecast clear");
    assert!(s.run("forecast query").contains("No forecast model"));
}

#[test]
fn forecast_simulation_statistics() {
    let mut s = Session::new();
    ts_data(&mut s, 60);
    s.script(
        "regress y L.y x
         estimates store m1
         tsappend, add(4)
         replace x = 0 if missing(x)
         forecast create m2
         forecast estimates m1
         forecast identity y2 = 2*y, generate
         forecast solve, begin(61) simulate(betas, statistic(mean, prefix(m_)) statistic(stddev, prefix(s_)) reps(400))",
    );
    // betas only: the mean path is close to the point forecast, spread grows
    let f = s.scalar("f_y[64]");
    assert!((s.scalar("m_y[64]") - f).abs() < 0.15, "mean {} vs {f}", s.scalar("m_y[64]"));
    assert!(s.scalar("s_y[64]") > 0.0);
    assert_close(s.scalar("f_y2[63]"), 2.0 * s.scalar("f_y[63]"), 1e-10);
    // errors with a zero-variance coefficient vector reproduce the error sd
    s.script(
        "matrix c = (0, 0, 1)
         matrix colnames c = q:L.y q:x q:_cons
         matrix E = (4)
         gen double q = y
         forecast create m3, replace
         forecast coefvector c, errorvariance(E)
         forecast solve, prefix(g_) begin(61) simulate(errors, statistic(stddev, prefix(sd_)) reps(2000))",
    );
    let sd = s.scalar("sd_q[62]");
    assert!((sd - 2.0).abs() < 0.15, "{sd}");
}

#[test]
fn fcast_and_varbasic_and_irf_add() {
    let mut s = Session::new();
    ts_data(&mut s, 80);
    s.run("var y x, lags(1/2)");
    let b = s.scalar("_b[y:L.y]");
    s.run("fcast compute f_, step(4)");
    assert!(s.has_var("f_y") && s.n() == 84);
    let mut s2 = Session::new();
    ts_data(&mut s2, 80);
    s2.run("varbasic y x, nograph");
    assert_close(s2.scalar("_b[y:L.y]"), b, 1e-10);
    s2.run("irf set r2, replace");
    s2.run("irf add first = varbasic, using(_varbasic)");
    let out = s2.run("irf describe");
    assert!(out.contains("first"), "{out}");
    s2.run("erase _varbasic.irf");
    s2.run("erase r2.irf");
}

#[test]
fn estat_acplot_ar1_autocorrelations() {
    let mut s = Session::new();
    s.script(
        "set obs 300
         set seed 2
         gen t = _n
         tsset t
         gen double y = 0
         replace y = 0.6*L.y + rnormal() in 2/l
         arima y, ar(1) nolog
         estat acplot, lags(4)
         matrix A = r(ac)",
    );
    let phi = s.scalar("_b[ARMA:L.ar]");
    assert_close(s.scalar("A[2,2]"), phi, 1e-6);
    assert_close(s.scalar("A[4,2]"), phi.powi(3), 1e-6);
    assert!(s.scalar("A[2,3]") < phi && s.scalar("A[2,4]") > phi);
}

fn dpd_data(s: &mut Session) {
    s.script(
        "set obs 400
         set seed 5
         gen id = ceil(_n/8)
         gen t = mod(_n-1,8)+1
         xtset id t
         gen double x = rnormal()
         gen double a = rnormal() if t==1
         bysort id (t): replace a = a[1]
         gen double y = a + rnormal() if t==1
         bysort id (t): replace y = 0.5*y[_n-1] + x + a + rnormal() if t>1",
    );
}

#[test]
fn xtdpd_reproduces_xtabond_and_xtdpdsys() {
    let mut s = Session::new();
    dpd_data(&mut s);
    s.run("xtabond y x");
    let (b, se) = (s.scalar("_b[L.y]"), s.scalar("_se[L.y]"));
    let out = s.run("xtdpd L(0/1).y x, dgmmiv(y) div(x)");
    assert!(out.contains("Dynamic panel-data estimation"), "{out}");
    assert_close(s.scalar("_b[L.y]"), b, 1e-9);
    assert_close(s.scalar("_se[L.y]"), se, 1e-9);
    s.run("xtdpdsys y x, twostep vce(robust)");
    let (b, se, bx) = (s.scalar("_b[L.y]"), s.scalar("_se[L.y]"), s.scalar("_b[x]"));
    s.run("xtdpd L(0/1).y x, dgmmiv(y) lgmmiv(y) div(x) twostep vce(robust)");
    assert_close(s.scalar("_b[L.y]"), b, 1e-9);
    assert_close(s.scalar("_b[x]"), bx, 1e-9);
    assert_close(s.scalar("_se[L.y]"), se, 1e-9);
    s.run("estat abond");
    s.run("estat sargan");
    // a limited lag range uses fewer instruments
    s.run("xtdpd L(0/1).y x, dgmmiv(y) div(x)");
    let j_all = s.scalar("e(zrank)");
    s.run("xtdpd L(0/1).y x, dgmmiv(y, lagrange(2 3)) div(x)");
    assert!(s.scalar("e(zrank)") < j_all);
}

fn coint_data(s: &mut Session) {
    s.script(
        "set obs 1200
         set seed 9
         gen id = ceil(_n/30)
         gen t = mod(_n-1,30)+1
         xtset id t
         gen double ex = rnormal()
         gen double ey = rnormal()
         bysort id (t): gen double x = sum(ex)
         gen double yc = 2 + 0.8*x + rnormal()
         bysort id (t): gen double yn = sum(ey)",
    );
}

#[test]
fn xtcointtest_rejects_for_cointegrated_panels_only() {
    let mut s = Session::new();
    coint_data(&mut s);
    for test in ["kao", "pedroni", "westerlund"] {
        s.run(&format!("xtcointtest {test} yc x"));
        s.run("matrix P = r(p)");
        assert!(s.scalar("P[1,1]") < 0.001, "{test}");
        s.run(&format!("xtcointtest {test} yn x"));
        s.run("matrix S = r(stats)");
        assert!(s.scalar("S[1,1]").abs() < 4.0, "{test} under the null");
    }
    s.run("xtcointtest kao yc x");
    assert_eq!(s.scalar("colsof(r(stats))"), 5.0);
    assert_eq!(s.scalar("r(N_g)"), 40.0);
    s.run("xtcointtest pedroni yn x, ar(same) trend");
    assert_eq!(s.scalar("colsof(r(stats))"), 3.0);
}

fn erm_data(s: &mut Session) {
    s.script(
        "set obs 500
         set seed 12
         gen double z = rnormal()
         gen double x = rnormal()
         gen double u = rnormal()
         gen double v = 0.5*u + rnormal()
         gen double w = 1 + z + v
         gen d = (z + v) > 0
         gen double y = 2 + x + 0.7*w + u
         gen double ys = 1 + x + d + u
         gen yb = ys > 1.5
         gen yo = 1 + (ys > 0.5) + (ys > 2)
         gen s = (0.5 + z + x + 0.6*u + rnormal()) > 0",
    );
}

#[test]
fn eregress_limits_match_regress_liml_and_heckman() {
    let mut s = Session::new();
    erm_data(&mut s);
    s.run("regress y x w");
    let b = s.scalar("_b[w]");
    s.run("eregress y x w");
    assert_close(s.scalar("_b[w]"), b, 1e-7);
    s.run("ivregress liml y x (w = z)");
    let (b, se) = (s.scalar("_b[w]"), s.scalar("_se[w]"));
    s.run("eregress y x, endogenous(w = z x)");
    assert_close(s.scalar("_b[y:w]"), b, 1e-6);
    assert_close(s.scalar("_se[y:w]"), se, 1e-4);
    s.run("gen double ysel = y if s");
    s.run("heckman ysel x w, select(s = z x w)");
    let (b, ll, rho) = (s.scalar("_b[ysel:w]"), s.scalar("e(ll)"), s.scalar("e(rho)"));
    s.run("eregress ysel x w, select(s = z x w)");
    assert_close(s.scalar("_b[ysel:w]"), b, 1e-6);
    assert_close(s.scalar("e(ll)"), ll, 1e-7);
    assert_close(s.scalar("_b[/corr(e.s,e.ysel)]"), rho, 1e-5);
    s.run("predict double xb1");
    assert!(s.has_var("xb1"));
}

#[test]
fn eprobit_eoprobit_eintreg_match_classic_estimators() {
    let mut s = Session::new();
    erm_data(&mut s);
    s.run("oprobit yo x");
    let (b, c1) = (s.scalar("_b[x]"), s.scalar("_b[/cut1]"));
    s.run("eoprobit yo x");
    assert_close(s.scalar("_b[yo:x]"), b, 1e-6);
    assert_close(s.scalar("_b[/cut1]"), c1, 1e-6);
    s.run("gen double lo = floor(ys)");
    s.run("gen double hi = ceil(ys)");
    s.run("replace lo = . in 1/20");
    s.run("intreg lo hi x");
    let (b, ll) = (s.scalar("_b[x]"), s.scalar("e(ll)"));
    s.run("eintreg lo hi x");
    assert_close(s.scalar("_b[lo:x]"), b, 1e-6);
    assert_close(s.scalar("e(ll)"), ll, 1e-8);
    s.run("biprobit (yb = x d) (d = z)");
    let (b, ll) = (s.scalar("_b[yb:d]"), s.scalar("e(ll)"));
    s.run("eprobit yb x, endogenous(d = z, probit)");
    assert_close(s.scalar("_b[yb:d]"), b, 1e-5);
    assert_close(s.scalar("e(ll)"), ll, 1e-8);
    s.run("gen ybs = yb if s");
    s.run("heckprobit ybs x, select(s = z x)");
    let ll = s.scalar("e(ll)");
    s.run("eprobit ybs x, select(s = z x)");
    assert_close(s.scalar("e(ll)"), ll, 1e-7);
}

#[test]
fn eregress_treatment_options() {
    let mut s = Session::new();
    erm_data(&mut s);
    // exogenous interacted treatment == regression with full interaction
    s.run("regress ys ibn.d ibn.d#c.x, noconstant");
    let b = s.scalar("_b[1.d#c.x]");
    s.run("eregress ys x, extreat(d)");
    assert_close(s.scalar("_b[ys:1.d#c.x]"), b, 1e-6);
    s.run("eregress ys x, entreat(d = z)");
    let r = s.scalar("_b[/corr(e.d,e.ys)]");
    assert!(r > 0.2 && r < 0.7, "{r}");
    s.run("eregress ys x, entreat(d = z, nointeract) vce(robust)");
    assert!(s.scalar("_b[ys:1.d]") > 0.5);
}

#[test]
fn xt_erm_match_mixed_and_xtprobit() {
    let mut s = Session::new();
    s.script(
        "set obs 240
         set seed 13
         gen id = ceil(_n/6)
         gen t = mod(_n-1,6)+1
         xtset id t
         gen double x = rnormal()
         gen double a = rnormal() if t==1
         bysort id (t): replace a = a[1]
         gen double y = 1 + x + a + rnormal()
         gen yb = (0.3 + x + 0.7*a + rnormal()) > 0",
    );
    s.run("mixed y x || id:, mle");
    let (b, ll) = (s.scalar("_b[x]"), s.scalar("e(ll)"));
    s.run("xteregress y x");
    assert_close(s.scalar("_b[y:x]"), b, 1e-5);
    assert_close(s.scalar("e(ll)"), ll, 1e-7);
    s.run("xtprobit yb x");
    let (b, ll) = (s.scalar("_b[x]"), s.scalar("e(ll)"));
    s.run("xteprobit yb x");
    assert_close(s.scalar("_b[yb:x]"), b, 1e-5);
    assert_close(s.scalar("e(ll)"), ll, 1e-6);
    // interval outcome with exact endpoints == linear model
    s.run("gen double y2 = y");
    s.run("xteintreg y y2 x");
    assert_close(s.scalar("e(ll)"), -s.scalar("-e(ll)"), 1e-12);
    let llx = s.scalar("e(ll)");
    s.run("xteregress y x");
    assert_close(s.scalar("e(ll)"), llx, 1e-6);
}

#[test]
fn xtheckman_runs_with_selection() {
    let mut s = Session::new();
    s.script(
        "set obs 180
         set seed 14
         gen id = ceil(_n/6)
         gen t = mod(_n-1,6)+1
         xtset id t
         gen double x = rnormal()
         gen double z = rnormal()
         gen double a = rnormal() if t==1
         bysort id (t): replace a = a[1]
         gen double c = rnormal() if t==1
         bysort id (t): replace c = c[1]
         gen double u = rnormal()
         gen s = (0.3 + z + x + 0.5*c + 0.5*u + rnormal()) > 0
         gen double y = 1 + x + a + u if s",
    );
    let out = s.run("xtheckman y x, select(s = z x)");
    assert!(out.contains("Random-effects regression with selection"), "{out}");
    let bx = s.scalar("_b[y:x]");
    assert!((bx - 1.0).abs() < 0.4, "{bx}");
    assert!(s.scalar("e(N_selected)") > 0.0);
}

#[test]
fn xtmlogit_binary_matches_clogit_and_xtlogit() {
    let mut s = Session::new();
    s.script(
        "set obs 320
         set seed 21
         gen id = ceil(_n/8)
         gen t = mod(_n-1,8)+1
         xtset id t
         gen double x = rnormal()
         gen double a = rnormal() if t==1
         bysort id (t): replace a = a[1]
         gen yb = (0.2 + x + a + logit(runiform())) > 0
         gen yo = yb + ((x + logit(runiform())) > 1)",
    );
    s.run("clogit yb x, group(id)");
    let b = s.scalar("_b[x]");
    s.run("xtmlogit yb x, fe baseoutcome(0)");
    assert_close(s.scalar("_b[1:x]"), b, 1e-6);
    s.run("xtlogit yb x, re");
    let (b, ll) = (s.scalar("_b[x]"), s.scalar("e(ll)"));
    s.run("xtmlogit yb x, re baseoutcome(0)");
    assert_close(s.scalar("_b[1:x]"), b, 1e-5);
    assert_close(s.scalar("e(ll)"), ll, 1e-6);
    s.run("xtmlogit yo x, fe");
    assert_eq!(s.scalar("e(k_out)"), 3.0);
}

#[test]
fn xtvar_recovers_panel_var_coefficients() {
    let mut s = Session::new();
    s.script(
        "set obs 6000
         set seed 31
         gen id = ceil(_n/12)
         gen t = mod(_n-1,12)+1
         xtset id t
         gen double a = rnormal() if t==1
         bysort id (t): replace a = a[1]
         gen double w = a + rnormal()
         gen double v = rnormal()
         bysort id (t): replace w = a + 0.5*w[_n-1] + 0.2*v[_n-1] + rnormal() if _n > 1
         bysort id (t): replace v = 0.3*v[_n-1] + 0.4*w[_n-1] + rnormal() if _n > 1",
    );
    s.run("xtvar w v, lags(1)");
    assert!((s.scalar("_b[w:L.w]") - 0.5).abs() < 0.1);
    assert!((s.scalar("_b[v:L.w]") - 0.4).abs() < 0.1);
    assert!((s.scalar("_b[v:L.v]") - 0.3).abs() < 0.1);
    assert_eq!(s.scalar("e(N_g)"), 500.0);
}

// ===== survival and multilevel tests (st_rest.rs, menl.rs) below this line =====

fn st_data(s: &mut Session, n: usize) {
    s.script(&format!(
        "set obs {n}
         set seed 11
         gen id = _n
         gen double x = rnormal()
         gen g = runiform() < .5
         gen double t = -ln(runiform())/exp(0.5*x + 0.4*g)
         gen double c = 2*runiform()
         gen d = t < c
         replace t = min(t, c)
         stset t, failure(d)"
    ));
}

#[test]
fn st_intcox_exact_endpoints_close_to_stcox() {
    let mut s = Session::new();
    st_data(&mut s, 120);
    s.run("stcox x g, breslow");
    let (bx, bg) = (s.scalar("_b[x]"), s.scalar("_b[g]"));
    s.run("gen double tl = t");
    s.run("gen double tu = cond(d, t, .)");
    let out = s.run("stintcox x g, interval(tl tu)");
    assert!(out.contains("interval-censored"), "{out}");
    // grouped-time NPMLE versus the partial likelihood: close, not identical
    assert!((s.scalar("_b[x]") - bx).abs() < 0.05, "{} vs {bx}", s.scalar("_b[x]"));
    assert!((s.scalar("_b[g]") - bg).abs() < 0.05);
    assert_eq!(s.scalar("e(N_rc)") + s.scalar("e(N_uncens)"), 120.0);
}

#[test]
fn st_intcox_wide_intervals_and_marginal_model() {
    let mut s = Session::new();
    st_data(&mut s, 100);
    s.run("gen double tl = floor(t*4)/4");
    s.run("gen double tu = cond(d, tl + .25, .)");
    s.run("replace tl = . if tl == 0");
    s.run("stintcox x g, interval(tl tu) vce(cluster id)");
    let (b, se) = (s.scalar("_b[x]"), s.scalar("_se[x]"));
    assert!(b > 0.0 && b < 1.2, "{b}");
    assert!(s.scalar("e(N_lc)") > 0.0 && s.scalar("e(N_int)") > 0.0);
    // one-equation marginal model = stintcox with subject clusters
    s.run("stmgintcox (tl tu = x g), id(id)");
    assert_close(s.scalar("_b[tu:x]"), b, 1e-6);
    assert_close(s.scalar("_se[tu:x]"), se, 1e-4);
}

#[test]
fn st_sttocc_full_risk_sets_reproduce_stcox() {
    let mut s = Session::new();
    st_data(&mut s, 80);
    s.run("stcox x g");
    let (bx, bg) = (s.scalar("_b[x]"), s.scalar("_b[g]"));
    s.run("sttocc, number(1000)");
    s.run("clogit _case x g, group(_set)");
    assert_close(s.scalar("_b[x]"), bx, 1e-6);
    assert_close(s.scalar("_b[g]"), bg, 1e-6);
}

#[test]
fn st_sttocc_samples_requested_controls() {
    let mut s = Session::new();
    st_data(&mut s, 80);
    s.run("count if d");
    let cases = s.scalar("r(N)");
    s.run("sttocc, number(2) match(g)");
    s.run("count if _case == 1");
    assert_eq!(s.scalar("r(N)"), cases);
    // controls share the case's matching value
    s.run("bysort _set (_case): gen bad = g != g[_N]");
    s.run("count if bad");
    assert_eq!(s.scalar("r(N)"), 0.0);
}

#[test]
fn st_sttoct_cttost_round_trip() {
    let mut s = Session::new();
    st_data(&mut s, 100);
    s.run("stcox g, breslow");
    let (b, ll) = (s.scalar("_b[g]"), s.scalar("e(ll)"));
    s.run("sttoct nf nc, by(g) replace");
    s.run("summarize nf");
    s.run("cttost");
    s.run("stcox g, breslow");
    assert_close(s.scalar("_b[g]"), b, 1e-8);
    assert_close(s.scalar("e(ll)"), ll, 1e-8);
}

#[test]
fn st_stmc_test_equals_logrank() {
    let mut s = Session::new();
    st_data(&mut s, 120);
    s.run("sts test g");
    let chi = s.scalar("r(chi2)");
    s.run("stmc g");
    assert_close(s.scalar("r(chi2)"), chi, 1e-8);
    assert!(s.scalar("r(lb)") < s.scalar("r(RR)") && s.scalar("r(RR)") < s.scalar("r(ub)"));
}

#[test]
fn st_stbase_collapses_split_records() {
    let mut s = Session::new();
    st_data(&mut s, 50);
    s.run("stset t, failure(d) id(id)");
    s.run("summarize _t");
    let tsum = s.scalar("r(sum)");
    s.run("stsplit sp, at(0.5 1)");
    assert!(s.ds().n_obs > 50);
    s.run("stbase, gap(gp)");
    assert_eq!(s.ds().n_obs, 50);
    s.run("summarize _t");
    assert_close(s.scalar("r(sum)"), tsum, 1e-10);
    s.run("summarize gp");
    assert_close(s.scalar("r(max)"), 0.0, 1e-10);
}

#[test]
fn st_gofplot_after_stcox_and_streg() {
    let mut s = Session::new();
    st_data(&mut s, 100);
    s.run("stcox x g");
    s.run("estat gofplot");
    assert!(s.scalar("r(H_max)") > 0.0);
    s.run("streg x g, dist(weibull)");
    s.run("estat gofplot");
    assert!(s.scalar("r(r_max)") > 0.0);
}

fn me_data(s: &mut Session) {
    s.script(
        "set obs 200
         set seed 3
         gen id = ceil(_n/5)
         gen t = mod(_n-1,5)+1
         gen double x = rnormal()
         gen double u = rnormal() if t == 1
         bysort id (t): replace u = u[1]
         gen double y = 1 + x + u + rnormal()",
    );
}

#[test]
fn menl_linear_model_equals_mixed_ml() {
    let mut s = Session::new();
    me_data(&mut s);
    s.run("mixed y x || id:");
    let (b, c, ll) = (s.scalar("_b[x]"), s.scalar("_b[_cons]"), s.scalar("e(ll)"));
    let vu = s.scalar("exp(2*_b[lns1_1_1:_cons])");
    let out = s.run("menl y = {b0} + {b1}*x + {U[id]}");
    assert!(out.contains("nonlinear regression"), "{out}");
    assert_close(s.scalar("_b[/b1]"), b, 1e-5);
    assert_close(s.scalar("_b[/b0]"), c, 1e-5);
    assert_close(s.scalar("e(ll)"), ll, 1e-7);
    assert_close(s.scalar("_b[/var(U)]"), vu, 1e-4);
    // linear-combination syntax gives the same fit
    s.run("menl y = {xb: x} + {U[id]}");
    assert_close(s.scalar("_b[xb:x]"), b, 1e-5);
}

#[test]
fn menl_nonlinear_recovers_parameters() {
    let mut s = Session::new();
    me_data(&mut s);
    s.run("gen double y2 = (5 + 0.5*u)*exp(-0.5*x) + 0.2*rnormal()");
    s.run("menl y2 = ({a=3} + {U[id]})*exp(-{b}*x)");
    assert!((s.scalar("_b[/a]") - 5.0).abs() < 0.3);
    assert!((s.scalar("_b[/b]") - 0.5).abs() < 0.05);
    assert!((s.scalar("_b[/var(U)]") - 0.25).abs() < 0.2);
    assert!((s.scalar("_b[/var(e)]") - 0.04).abs() < 0.02);
}

#[test]
fn menl_estat_reports_after_mixed() {
    let mut s = Session::new();
    me_data(&mut s);
    s.run("mixed y x || id:");
    let vu = s.scalar("exp(2*_b[lns1_1_1:_cons])");
    let ve = s.scalar("e(sigma_e)^2");
    s.run("estat sd");
    s.run("matrix S = r(b)");
    assert_close(s.scalar("S[1,1]^2"), vu, 1e-8);
    s.run("estat wcorrelation");
    s.run("matrix C = r(Corr)");
    assert_close(s.scalar("C[2,1]"), vu / (vu + ve), 1e-8);
    s.run("estat df");
    s.run("matrix D = r(df)");
    assert_eq!(s.scalar("D[1,1]"), 198.0);
}
