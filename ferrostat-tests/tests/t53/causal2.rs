//! lateffects / latebalance / lateoverlap, cate, telasso, mediate and
//! stteffects: closed forms and agreement with other estimators.

#[allow(unused_imports)]
use ferrostat_tests::*;

fn iv_data(s: &mut Session) {
    s.script(
        "clear
         set obs 600
         set seed 4242
         gen x = rnormal()
         gen z = runiform() < .5
         gen u = rnormal()
         gen d = runiform() < .15 + .55*z + .1*(u > 0)
         gen y = 1 + 2*d + .5*u + rnormal()",
    );
}

#[test]
fn late_without_covariates_is_the_wald_ratio() {
    let mut s = Session::new();
    iv_data(&mut s);
    s.run("ivregress 2sls y (d = z), vce(robust)");
    let b_iv = s.scalar("_b[d]");
    let se_iv = s.scalar("_se[d]");
    for m in ["ra", "ipw", "aipw", "ipwra"] {
        s.run(&format!("lateffects {m} (y) (d) (z)"));
        assert_close(s.scalar("_b[LATE:r1vs0.d]"), b_iv, 1e-7);
        // M-estimation sandwich vs HC1-type IV robust: equal up to n/(n-k)
        let se = s.scalar("_se[LATE:r1vs0.d]");
        assert!((se / se_iv - 1.0).abs() < 0.01, "{m}: {se} vs {se_iv}");
    }
    // complier means: their difference is the LATE
    s.run("lateffects ra (y) (d) (z), pomeans");
    let c1 = s.scalar("_b[cPOmeans:1.d]");
    let c0 = s.scalar("_b[cPOmeans:0.d]");
    assert_close(c1 - c0, b_iv, 1e-7);
    s.fails("lateffects ra (y x) (d) (z x)");
}

#[test]
fn late_with_covariates_and_balance_tools() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 800
         set seed 99
         gen x = rnormal()
         gen z = runiform() < invlogit(.3*x)
         gen d = runiform() < .1 + .6*z
         gen y = 1 + 1.5*d + x + rnormal()",
    );
    s.run("lateffects aipw (y x) (d x) (z x)");
    let late = s.scalar("_b[LATE:r1vs0.d]");
    assert!((late - 1.5).abs() < 0.4, "{late}");
    s.run("lateffects ipwra (y x) (d x) (z x)");
    assert!((s.scalar("_b[LATE:r1vs0.d]") - late).abs() < 0.2);
    let out = s.run("latebalance summarize");
    assert!(out.contains("Standardized diff."), "{out}");
    // weighting balances the covariate
    assert!(s.scalar("abs(el(r(table),1,2))") < s.scalar("abs(el(r(table),1,1))"));
    s.run("latebalance overid");
    assert!(s.scalar("r(df)") == 1.0);
    s.run("latebalance density x");
    s.run("lateoverlap");
}

#[test]
fn cate_recovers_constant_and_linear_effects() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 2000
         set seed 7
         gen x = rnormal()
         gen w = rnormal()
         gen t = runiform() < invlogit(.4*x - .3*w)
         gen y = 1 + x + .5*w + 2*t + rnormal()",
    );
    s.run("cate po (y x) (t), controls(w) rseed(3)");
    assert!((s.scalar("_b[ATE:r1vs0.t]") - 2.0).abs() < 0.15);
    assert!(s.scalar("abs(el(e(b_cate),1,1))") < 0.2);
    s.run("estat heterogeneity");
    assert!(s.scalar("r(p)") > 0.001);
    // heterogeneous effect 2 + x
    s.run("replace y = y + t*x");
    for m in ["po", "aipw"] {
        s.run(&format!("cate {m} (y x) (t), controls(w) rseed(3)"));
        let slope = s.scalar("el(e(b_cate),1,1)");
        assert!((slope - 1.0).abs() < 0.15, "{m}: {slope}");
        assert!((s.scalar("_b[ATE:r1vs0.t]") - 2.0).abs() < 0.15);
    }
    s.run("estat projection x");
    assert!((s.scalar("el(r(b),1,1)") - 1.0).abs() < 0.2);
    s.run("predict tau");
    s.run("summarize tau");
    assert!(s.scalar("r(N)") == 2000.0);
    s.run("predict ps, psi");
    s.run("estat ate");
    s.run("cate po (y x) (t), controls(w) omethod(lasso) tmethod(lasso) rseed(3)");
    assert!((s.scalar("el(e(b_cate),1,1)") - 1.0).abs() < 0.2);
    s.fails("cate po (y x) (t), omethod(rforest)");
}

#[test]
fn telasso_matches_aipw_when_all_covariates_are_selected() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 1500
         set seed 21
         gen x1 = rnormal()
         gen x2 = rnormal()
         gen t = runiform() < invlogit(.8*x1 - .8*x2)
         gen y = 1 + 2*x1 - 2*x2 + 1.5*t + rnormal()",
    );
    s.run("telasso (y x1 x2) (t x1 x2)");
    assert_close(s.scalar("e(k_othvars_sel)"), 2.0, 0.0);
    assert_close(s.scalar("e(k_tvars_sel)"), 2.0, 0.0);
    let b = s.scalar("_b[ATE:r1vs0.t]");
    let se = s.scalar("_se[ATE:r1vs0.t]");
    let p0 = s.scalar("_b[POmean:0.t]");
    s.run("teffects aipw (y x1 x2) (t x1 x2)");
    assert_close(b, s.scalar("_b[ATE:r1vs0.t]"), 1e-6);
    assert_close(p0, s.scalar("_b[POmean:0.t]"), 1e-6);
    assert!((se / s.scalar("_se[ATE:r1vs0.t]") - 1.0).abs() < 1e-3);
    // noise covariates are dropped
    s.run("gen n1 = rnormal()");
    s.run("telasso (y x1 x2 n1) (t x1 x2 n1)");
    assert!(!s.run("display \"`e(othvars_sel)'\"").contains("n1"));
}

#[test]
fn mediate_linear_effects_are_products_of_coefficients() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 500
         set seed 5
         gen x = rnormal()
         gen t = runiform() < .5
         gen m = .5 + .8*t + .3*x + rnormal()
         gen y = 1 + .7*t + 1.2*m + .4*x + rnormal()",
    );
    s.run("regress m t x");
    let a = s.scalar("_b[t]");
    s.run("regress y t m x");
    let (c, b) = (s.scalar("_b[t]"), s.scalar("_b[m]"));
    s.run("mediate (y x) (m x) (t), nointeraction");
    assert_close(s.scalar("_b[NIE:r1vs0.t]"), a * b, 1e-7);
    assert_close(s.scalar("_b[NDE:r1vs0.t]"), c, 1e-7);
    assert_close(s.scalar("_b[TE:r1vs0.t]"), a * b + c, 1e-7);
    s.run("estat proportion");
    assert_close(s.scalar("r(prop)"), a * b / (a * b + c), 1e-7);
    // with the interaction: NDE = c + d * E[M(0)]
    s.run("gen tm = t*m");
    s.run("regress y t m tm x");
    let (c2, b2, d2) = (s.scalar("_b[t]"), s.scalar("_b[m]"), s.scalar("_b[tm]"));
    s.run("mediate (y x) (m x) (t)");
    s.run("quietly regress m t x");
    s.run("predict m0hat, xb");
    s.run("replace m0hat = m0hat - _b[t]*t");
    s.run("summarize m0hat");
    let em0 = s.scalar("r(mean)");
    s.run("mediate (y x) (m x) (t)");
    assert_close(s.scalar("_b[NDE:r1vs0.t]"), c2 + d2 * em0, 1e-6);
    assert_close(s.scalar("_b[NIE:r1vs0.t]"), (b2 + d2) * a, 1e-6);
    // binary mediator, logit outcome and a three-level treatment run
    s.run("gen mb = m > 1");
    s.run("gen yb = y > 2");
    s.run("mediate (yb x, logit) (mb x, probit) (t)");
    s.run("gen t3 = t + (x > 1)");
    let out = s.run("mediate (y x) (m x) (t3), nie");
    assert!(out.contains("(2 vs 0)"), "{out}");
}

#[test]
fn stteffects_recover_exponential_mean_times() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 3000
         set seed 13
         gen x = runiform() < .5
         gen t = runiform() < .3 + .3*x
         gen double lam = exp(-1 + .5*x - .7*t)
         gen double ft = -ln(runiform())/lam
         gen double ct = -ln(runiform())/.05
         gen double tm = min(ft, ct)
         gen byte fail = ft <= ct
         stset tm, failure(fail)",
    );
    // true ATE: mean over x of exp(1 - .5x)(exp(.7) - 1)
    let truth = 0.5 * (1.0f64.exp() + 0.5f64.exp()) * (0.7f64.exp() - 1.0);
    s.run("stteffects ra (x, exponential) (t)");
    let ra = s.scalar("_b[ATE:r1vs0.t]");
    assert!((ra - truth).abs() < 0.12 * truth, "{ra} vs {truth}");
    s.run("stteffects ipwra (x, exponential) (t x)");
    let ipwra = s.scalar("_b[ATE:r1vs0.t]");
    assert!((ipwra - ra).abs() < 0.02 * truth, "{ipwra} vs {ra}");
    s.run("stteffects ra (x) (t)");
    assert!((s.scalar("_b[ATE:r1vs0.t]") - truth).abs() < 0.15 * truth);
    s.run("stteffects ipw (t x) (x)");
    assert!((s.scalar("_b[ATE:r1vs0.t]") - truth).abs() < 0.2 * truth);
    s.run("stteffects wra (x) (t) (x)");
    assert!((s.scalar("_b[ATE:r1vs0.t]") - truth).abs() < 0.2 * truth);
    s.run("stteffects ra (x) (t), pomeans");
    assert!(s.scalar("_b[POmeans:1.t]") > s.scalar("_b[POmeans:0.t]"));
    s.run("stteffects ra (x) (t), atet");
    s.fails("stteffects ipw (t) (x) (x)");
}
