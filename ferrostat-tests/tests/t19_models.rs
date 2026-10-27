//! Count, robust, multivariate and multilevel models: zip/zinb, rreg,
//! manova/manovatest/mvreg, melogit, `separate`, `estat gof` after poisson
//! and constant-only starting values for binary/multinomial iteration logs.
//!
//! Reference values were computed independently (direct maximization of the
//! textbook likelihoods, determinant ratios for Wilks' lambda, a separate
//! adaptive-quadrature evaluation for the random-intercept logit).

use ferrostat_tests::*;

fn zip_data(s: &mut Session) {
    s.script(
        "clear
         set obs 240
         gen child = mod(_n, 4)
         gen camper = mod(_n, 3) == 0
         gen persons = 1 + mod(_n*7, 4)
         gen u = mod(_n*37, 101)/101
         gen lam = exp(1.5 - 0.9*child + 0.8*camper)
         gen count = cond(u < invlogit(1.2 - 0.6*persons), 0, floor(lam*(0.5 + mod(_n*53, 97)/97)))
         gen countnb = cond(u < invlogit(1.2 - 0.6*persons), 0, floor(lam*(-ln(1 - mod(_n*53, 97)/97.5))*1.5))",
    );
}

fn first_iteration(out: &str) -> f64 {
    let line = out.lines().find(|l| l.starts_with("Iteration 0:")).expect("iteration log");
    numbers(line.split('=').nth(1).unwrap())[0]
}

#[test]
fn iteration_zero_is_constant_only_model() {
    let mut s = Session::new();
    s.script(
        "set obs 60
         gen x = mod(_n*7, 11)
         gen z = mod(_n*5, 3)
         gen y = (mod(_n*13, 17) + x) > 12
         gen m = 1 + mod(_n*7, 3) + (x > 7)",
    );
    // ll of the constant-only model: n1 ln p + n0 ln(1-p)
    s.run("count if y");
    let n1 = s.scalar("r(N)");
    let n = 60.0;
    let p = n1 / n;
    let ll0 = n1 * p.ln() + (n - n1) * (1.0 - p).ln();
    for cmd in ["logit y x z", "probit y x z"] {
        let out = s.run(cmd);
        assert_close(first_iteration(&out), ll0, 1e-7);
        assert_close(s.scalar("e(ll_0)"), ll0, 1e-9);
    }
    let out = s.run("mlogit m x");
    assert_close(first_iteration(&out), s.scalar("e(ll_0)"), 1e-7);
    let out = s.run("ologit m x");
    assert_close(first_iteration(&out), s.scalar("e(ll_0)"), 1e-7);
}

#[test]
fn estat_gof_after_poisson() {
    let mut s = Session::new();
    s.script(
        "set obs 80
         gen x = mod(_n*7, 11)/5
         gen g = 1 + mod(_n, 3)
         gen y = floor(exp(0.2 + 0.4*x)*(0.3 + mod(_n*31, 23)/15))",
    );
    s.run("poisson y x i.g");
    let out = s.run("estat gof");
    assert!(out.contains("Deviance goodness-of-fit ="), "{out}");
    assert!(out.contains("Prob > chi2(76)"), "{out}");
    s.script(
        "predict double mu
         gen dv = 2*(cond(y > 0, y*ln(y/mu), 0) - (y - mu))
         gen pe = (y - mu)^2/mu
         quietly summarize dv",
    );
    let dev = s.scalar("r(sum)");
    s.run("quietly summarize pe");
    let pear = s.scalar("r(sum)");
    s.run("poisson y x i.g");
    s.run("estat gof");
    assert_close(s.scalar("r(chi2_d)"), dev, 1e-8);
    assert_close(s.scalar("r(chi2_p)"), pear, 1e-8);
    assert_eq!(s.scalar("r(df)"), 76.0);
    assert_close(s.scalar("r(p_d)"), s.scalar("chi2tail(76, r(chi2_d))"), 1e-10);
}

#[test]
fn zip_matches_direct_maximization() {
    let mut s = Session::new();
    zip_data(&mut s);
    let out = s.run("zip count child camper, inflate(persons) vuong");
    assert!(out.contains("Zero-inflated Poisson regression"), "{out}");
    assert!(out.contains("Vuong test of zip vs. standard Poisson:"), "{out}");
    assert_close(s.scalar("e(ll)"), -216.8017496697241, 1e-9);
    let expect = [
        ("count:child", -1.3110667653244468, 0.09966323873495335),
        ("count:camper", 1.0646689761370343, 0.1336683492768065),
        ("count:_cons", 1.2943909888361553, 0.11752804361944588),
        ("inflate:persons", -1.0379913691838436, 0.28407912416477743),
        ("inflate:_cons", 1.6252596915894761, 0.45149381850398523),
    ];
    for (name, b, se) in expect {
        assert_close(s.scalar(&format!("_b[{name}]")), b, 1e-6);
        assert_close(s.scalar(&format!("_se[{name}]")), se, 1e-5);
    }
    assert_eq!(s.scalar("e(N)"), 240.0);
    assert_eq!(s.scalar("e(N_zero)"), 156.0);
    assert_eq!(s.scalar("e(df_m)"), 2.0);
    assert_close(s.scalar("e(chi2)"), 2.0 * (s.scalar("e(ll)") - s.scalar("e(ll_0)")), 1e-9);
    // predicted counts: (1 - Pr(inflate)) * exp(xb)
    s.run("predict n");
    s.run("predict pz, pr");
    s.run("gen chk = (1 - invlogit(_b[inflate:_cons] + _b[inflate:persons]*persons))*exp(_b[count:_cons] + _b[count:child]*child + _b[count:camper]*camper)");
    let n = s.nums("n");
    let chk = s.nums("chk");
    for (a, b) in n.iter().zip(&chk) {
        assert_close(*a, *b, 1e-6);
    }
    // robust variance and factor variables
    let out = s.run("zip count child i.camper, inflate(persons) vce(robust)");
    assert!(out.contains("Log pseudolikelihood"), "{out}");
    assert!(out.contains("Wald chi2(2)"), "{out}");
    assert_close(s.scalar("_b[count:1.camper]"), 1.0646689761370343, 1e-6);
}

#[test]
fn zinb_matches_direct_maximization() {
    let mut s = Session::new();
    zip_data(&mut s);
    let out = s.run("zinb countnb child camper, inflate(persons)");
    assert!(out.contains("/lnalpha"), "{out}");
    assert!(row(&out, "alpha").is_some(), "{out}");
    assert_close(s.scalar("e(ll)"), -307.3290352155853, 1e-8);
    let expect = [
        ("countnb:child", -1.170237685372435, 0.14810882349080262),
        ("countnb:camper", 1.0287161371099012, 0.2359654240959371),
        ("countnb:_cons", 1.8247472551786916, 0.2167748293876976),
        ("inflate:persons", -0.6695112747388106, 0.23664367597767177),
        ("inflate:_cons", 1.2552522763848077, 0.4216077967379959),
    ];
    for (name, b, se) in expect {
        assert_close(s.scalar(&format!("_b[{name}]")), b, 1e-5);
        assert_close(s.scalar(&format!("_se[{name}]")), se, 1e-4);
    }
    assert_close(s.scalar("e(alpha)"), (-0.10538301598562484f64).exp(), 1e-4);
}

#[test]
fn rreg_huber_then_biweight() {
    let mut s = Session::new();
    s.script(
        "set obs 40
         gen x1 = mod(_n*7, 11)
         gen x2 = mod(_n*5, 13)/2
         gen y = 3 + 2*x1 - x2 + (mod(_n*17, 19) - 9)/3
         replace y = y + 40 in 5
         replace y = y - 25 in 17
         replace x1 = 30 in 40
         replace y = 10 in 40",
    );
    let out = s.run("rreg y x1 x2, gen(w)");
    assert!(out.contains("   Huber iteration 1:  maximum difference in weights = "), "{out}");
    assert!(out.contains("Biweight iteration"), "{out}");
    assert!(out.contains("Robust regression"), "{out}");
    // observation 40 has Cook's D > 1 and is dropped
    assert_eq!(s.scalar("e(N)"), 39.0);
    assert_eq!(s.scalar("e(df_r)"), 36.0);
    assert_eq!(s.scalar("missing(w[40])"), 1.0);
    let b = [2.0101403817756474, -1.0601273643574256, 3.2466187122226295];
    let se = [0.11085538011196854, 0.18683694743255352, 0.8334496011891219];
    for (i, nm) in ["x1", "x2", "_cons"].iter().enumerate() {
        assert_close(s.scalar(&format!("_b[{nm}]")), b[i], 1e-7);
        assert_close(s.scalar(&format!("_se[{nm}]")), se[i], 1e-6);
    }
    // the gross outliers are strongly downweighted
    assert!(s.scalar("w[5]") < 0.05);
    assert!(s.scalar("w[1]") > 0.5);
}

#[test]
fn manova_statistics_and_manovatest() {
    let mut s = Session::new();
    s.script(
        "set obs 24
         gen grp = 1 + mod(_n, 3)
         gen y1 = grp + mod(_n*7, 5)/2
         gen y2 = 2*(grp==2) + mod(_n*5, 7)/3
         gen y3 = mod(_n*11, 13)/4 - (grp==3)",
    );
    let out = s.run("manova y1 y2 y3 = grp");
    assert!(out.contains("W = Wilks' lambda      L = Lawley-Hotelling trace"), "{out}");
    assert!(out.contains("e = exact, a = approximate, u = upper bound on F"), "{out}");
    let w = row(&out, "grp").unwrap();
    assert_close(w[0], 0.1068, 1e-9);
    // Wilks' lambda = det(E) / det(E + H)
    assert_close(s.scalar("el(e(stat_1),1,1)"), 0.10678641658100499, 1e-7);
    assert_eq!(s.scalar("e(df_r)"), 21.0);
    // Wilks exact F for q = 2
    let lam = 0.10678641658100499f64;
    let f = (1.0 - lam.sqrt()) / lam.sqrt() * 38.0 / 6.0;
    assert_close(s.scalar("el(e(stat_1),2,1)"), f, 1e-6);
    let out = s.run("manovatest, showorder");
    assert!(out.contains("1: (grp==1)") && out.contains("4: _cons"), "{out}");
    s.run("matrix c = (0, 1, -1, 0)");
    let out = s.run("manovatest, test(c)");
    assert!(out.contains("2.grp - 3.grp = 0"), "{out}");
    assert_close(s.scalar("el(r(stat),1,1)"), 0.3087889968361139, 1e-7);
    // one-degree-of-freedom hypothesis: P = 1 - W, L = R = P / W, all F equal
    assert_close(s.scalar("el(r(stat),1,2)"), 1.0 - 0.3087889968361139, 1e-7);
    assert_close(s.scalar("el(r(stat),1,4)"), (1.0 - 0.3087889968361139) / 0.3087889968361139, 1e-6);
    assert_close(s.scalar("el(r(stat),2,1)"), s.scalar("el(r(stat),2,4)"), 1e-9);
    // mvreg: equation-by-equation least squares
    s.run("mvreg y1 y2 y3 = i.grp");
    let b = s.scalar("_b[y2:2.grp]");
    s.run("regress y2 i.grp");
    assert_close(b, s.scalar("_b[2.grp]"), 1e-10);
}

#[test]
fn melogit_random_intercept() {
    let mut s = Session::new();
    s.script(
        "set obs 480
         gen g = ceil(_n/12)
         gen x = mod(_n*13, 17)/4 - 2
         gen v = (mod(_n*29, 97) + 0.5)/97
         gen y = ((mod(g*7, 11) - 5)/2.5 + 0.8*x - 0.5 + ln(v/(1-v))) > 0",
    );
    let out = s.run("melogit y x || g:");
    assert!(out.contains("Mixed-effects logistic regression"), "{out}");
    assert!(out.contains("Integration method: mvaghermite"), "{out}");
    assert!(out.contains("var(_cons)|"), "{out}");
    assert!(out.contains("LR test vs. logistic model: chibar2(01) ="), "{out}");
    assert_eq!(s.scalar("e(N)"), 480.0);
    assert_eq!(s.scalar("el(e(N_g),1,1)"), 40.0);
    assert_close(s.scalar("e(ll)"), -277.354523, 1e-8);
    assert_close(s.scalar("_b[y:x]"), 0.7400105, 1e-5);
    assert_close(s.scalar("el(e(b),1,3)"), 1.22319, 1e-4);
    // many quadrature points: agrees with an independent evaluation of the
    // marginal likelihood (40-point quadrature at the mode)
    s.run("melogit y x || g:, intpoints(30)");
    assert_close(s.scalar("e(ll)"), -277.35450787, 1e-8);
    assert_close(s.scalar("_b[y:_cons]"), -0.4636228, 1e-5);
    // replay with odds ratios
    let out = s.run("melogit, or");
    assert!(out.contains("Odds ratio"), "{out}");
    let r = row(&out, "x").unwrap();
    assert_close(r[0], s.scalar("exp(_b[y:x])"), 1e-6);
    // meqrlogit is accepted as the same model
    s.run("meqrlogit y x || g:, intpoints(30)");
    assert_close(s.scalar("e(ll)"), -277.35450787, 1e-8);
}

#[test]
fn separate_by_variable_and_expression() {
    let mut s = Session::new();
    s.script(
        "set obs 6
         gen v = _n * 10
         gen g = 1 + mod(_n, 3)
         label define gl 1 \"low\" 2 \"mid\" 3 \"high\"
         label values g gl",
    );
    let out = s.run("separate v, by(g)");
    assert!(out.contains("v, g == low"), "{out}");
    assert_eq!(s.nums("v1")[2], 30.0);
    assert!(s.nums("v1")[0].is_nan());
    assert_eq!(s.nums("v2")[0], 10.0);
    let out = s.run("separate v, by(v > 25) gen(hi) veryshortlabel");
    assert!(out.contains("hi0") && out.contains("hi1"), "{out}");
    assert_eq!(s.nums("hi1")[2], 30.0);
    assert!(s.nums("hi1")[1].is_nan());
    assert_eq!(s.nums("hi0")[1], 20.0);
    s.fails("separate v, by(g)");
}
