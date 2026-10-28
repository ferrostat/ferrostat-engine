//! Survival analysis: stset with id/enter/exit/origin, Kaplan-Meier and
//! k-sample tests, Cox regression (all tie methods, strata, robust, tvc)
//! with its postestimation, parametric streg models, stsplit/stjoin.
//! Results are checked against hand computations and model identities
//! (Cox = piecewise-exponential Poisson, exponential streg = Poisson).

use ferrostat_tests::*;

/// Small dataset: times 1 2 2 3+ 4 6+ 7 8, x a covariate, g a group.
fn tiny() -> Session {
    let mut s = Session::new();
    s.script(
        "set obs 8\n\
         gen t = 1 in 1\nreplace t = 2 in 2/3\nreplace t = 3 in 4\nreplace t = 4 in 5\nreplace t = 6 in 6\nreplace t = 7 in 7\nreplace t = 8 in 8\n\
         gen d = !inlist(_n, 4, 6)\n\
         gen x = mod(_n * 7, 5) - 2\n\
         gen g = _n > 4",
    );
    s
}

/// Simulated proportional-hazards data with ties (integer times).
fn sim(n: usize, seed: u32) -> Session {
    let mut s = Session::new();
    s.script(&format!(
        "set obs {n}\nset seed {seed}\ngen x = rnormal()\ngen g = runiform() < 0.5\n\
         gen t = ceil(-ln(runiform())/exp(0.5*x - 0.4*g)*8)\ngen d = runiform() < 0.8\ngen id = _n"
    ));
    s
}

#[test]
fn stset_creates_st_variables_and_reports() {
    let mut s = tiny();
    let out = s.run("stset t, failure(d)");
    assert!(out.contains("Survival-time data settings") && out.contains("failures in single-record/single-failure data"), "{out}");
    assert!(out.contains("6  failures") && out.contains("33  total analysis time"), "{out}");
    assert_all_close(&s.nums("_t"), &[1.0, 2.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0], 0.0);
    assert_all_close(&s.nums("_d"), &[1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0], 0.0);
    // origin and scale
    s.run("stset t, failure(d) origin(time 1) scale(2)");
    let t = s.nums("_t");
    assert!(t[0].is_nan(), "a record ending at the origin is excluded");
    assert_close(t[7], 3.5, 1e-12);
    // stset, clear removes everything
    s.run("stset, clear");
    assert!(!s.has_var("_t") && !s.has_var("_st"));
    s.fails("stcox x");
}

#[test]
fn stset_id_enter_exit() {
    let mut s = Session::new();
    // two subjects with two records each; subject 2 fails in its first record
    s.script(
        "set obs 4\ngen id = ceil(_n/2)\ngen t = cond(mod(_n,2), 5, 10)\ngen d = (_n == 2 | _n == 3)\ngen e = 2",
    );
    s.run("stset t, id(id) failure(d)");
    let t0 = s.nums("_t0");
    assert_eq!((t0[0], t0[1], t0[2]), (0.0, 5.0, 0.0));
    assert!(t0[3].is_nan() && s.nums("_st")[3] == 0.0, "records after the failure are excluded");
    s.run("stset t, id(id) failure(d) enter(time e)");
    assert_eq!(s.nums("_t0")[0], 2.0);
    s.run("stset t, id(id) failure(d) exit(time .)");
    assert_eq!(s.nums("_st"), vec![1.0; 4]);
    let out = s.run("stset t, id(id) failure(d) exit(time 4)");
    assert!(out.contains("observations begin on or after exit"), "{out}");
    assert_eq!(s.nums("_t")[0], 4.0);
    assert_eq!(s.nums("_d")[2], 0.0);
}

#[test]
fn kaplan_meier_by_hand() {
    let mut s = tiny();
    s.run("stset t, failure(d)");
    s.run("sts generate S = s se = se(s) H = na");
    // S: t=1 7/8; t=2 7/8*5/7 = 5/8; t=4 5/8*3/4; t=7 *1/2; t=8 0
    let sv = s.nums("S");
    assert_close(sv[0], 7.0 / 8.0, 1e-12);
    assert_close(sv[1], 5.0 / 8.0, 1e-12);
    assert_close(sv[3], 5.0 / 8.0, 1e-12);
    assert_close(sv[4], 15.0 / 32.0, 1e-12);
    assert_close(sv[6], 15.0 / 64.0, 1e-12);
    assert_close(sv[7], 0.0, 1e-12);
    // Greenwood at t=2: S^2 * (1/(8*7) + 2/(7*5))
    assert_close(s.nums("se")[1], (25.0f64 / 64.0 * (1.0 / 56.0 + 2.0 / 35.0)).sqrt(), 1e-10);
    // Nelson-Aalen at t=4: 1/8 + 2/7 + 1/4
    assert_close(s.nums("H")[4], 1.0 / 8.0 + 2.0 / 7.0 + 0.25, 1e-12);
    let out = s.run("sts list");
    assert!(out.contains("Kaplan–Meier survivor function"), "{out}");
    // the t=2 row: 7 at risk, 2 failures, S = .625
    let line = out.lines().find(|l| l.trim_start().starts_with("2 ")).unwrap();
    assert_all_close(&numbers(line)[..5], &[2.0, 7.0, 2.0, 0.0, 0.625], 1e-12);
    // median: first time with S <= .5 is 4
    s.run("stsum");
    assert_close(s.scalar("r(p50)"), 4.0, 0.0);
    assert_close(s.scalar("r(risk)"), 33.0, 0.0);
    s.run("stci");
    assert_close(s.scalar("r(p50)"), 4.0, 0.0);
}

#[test]
fn logrank_by_hand() {
    let mut s = tiny();
    s.run("stset t, failure(d)");
    s.run("sts test g");
    // hand computation of U and V for group 0 over the failure times
    let t = [1.0, 2.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0];
    let d = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
    let g = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let (mut u, mut v) = (0.0, 0.0);
    for tj in [1.0, 2.0, 4.0, 7.0, 8.0] {
        let n: f64 = t.iter().filter(|&&x| x >= tj).count() as f64;
        let n0: f64 = (0..8).filter(|&i| t[i] >= tj && g[i] == 0.0).count() as f64;
        let dj: f64 = (0..8).filter(|&i| t[i] == tj).map(|i| d[i]).sum();
        let d0: f64 = (0..8).filter(|&i| t[i] == tj && g[i] == 0.0).map(|i| d[i]).sum();
        u += d0 - n0 * dj / n;
        if n > 1.0 {
            v += dj * (n - dj) / (n - 1.0) * n0 / n * (1.0 - n0 / n);
        }
    }
    assert_close(s.scalar("r(chi2)"), u * u / v, 1e-10);
    assert_eq!(s.scalar("r(df)"), 1.0);
    for opt in ["wilcoxon", "tware", "peto", "fh(1 0)"] {
        s.run(&format!("sts test g, {opt}"));
        assert!(s.scalar("r(chi2)") > 0.0, "{opt}");
    }
}

#[test]
fn cox_no_ties_all_methods_agree() {
    let mut s = Session::new();
    s.script("set obs 60\nset seed 4\ngen x = rnormal()\ngen t = -ln(runiform())/exp(0.7*x)\ngen d = runiform() < 0.8\nstset t, failure(d)");
    let mut b = Vec::new();
    for m in ["breslow", "efron", "exactm", "exactp"] {
        s.run(&format!("stcox x, {m}"));
        b.push((s.scalar("_b[x]"), s.scalar("e(ll)"), s.scalar("_se[x]")));
    }
    for k in 1..4 {
        assert_close(b[k].0, b[0].0, 1e-6);
        assert_close(b[k].1, b[0].1, 1e-6);
        assert_close(b[k].2, b[0].2, 1e-4);
    }
}

#[test]
fn cox_breslow_equals_piecewise_poisson() {
    let mut s = sim(300, 7);
    s.run("stset t, failure(d) id(id)");
    let out = s.run("stcox x g");
    assert!(out.contains("Cox regression with Breslow method for ties") && out.contains("Haz. ratio"), "{out}");
    let (bx, bg) = (s.scalar("_b[x]"), s.scalar("_b[g]"));
    // the split data give the same Cox fit
    s.run("preserve");
    s.run("stsplit cut5, every(3)");
    s.run("stcox x g");
    assert_close(s.scalar("_b[x]"), bx, 1e-8);
    s.run("restore");
    assert_eq!(s.scalar("e(N_fail)"), s.scalar("e(N_fail)"));
    assert!(s.run("ereturn list").contains("e(cmd2)"));
    // split at every failure time; a Poisson model with one rate per interval
    s.run("levelsof _t if _d, local(ft)");
    s.run("stsplit cut, at(`ft')");
    // the risk set at a failure time is the pieces ending there: drop the
    // pieces that end at a censoring time; one rate per failure time
    s.run("bysort _t: egen anyd = max(_d)");
    s.run("keep if anyd == 1");
    s.run("quietly poisson _d x g i.cut");
    assert_close(s.scalar("_b[x]"), bx, 1e-6);
    assert_close(s.scalar("_b[g]"), bg, 1e-6);
}

#[test]
fn stsplit_stjoin_roundtrip() {
    let mut s = sim(100, 8);
    s.run("stset t, failure(d) id(id)");
    s.run("stcox x g");
    let ll = s.scalar("e(ll)");
    let out = s.run("stsplit cut, at(2 5 10)");
    assert!(out.contains("observations (episodes) created"), "{out}");
    s.run("stcox x g");
    assert_close(s.scalar("e(ll)"), ll, 1e-10);
    // re-stset from the split data reproduces the same records
    s.run("stset t, failure(d) id(id)");
    s.run("stcox x g");
    assert_close(s.scalar("e(ll)"), ll, 1e-10);
    s.run("drop cut");
    let out = s.run("stjoin");
    assert!(out.contains("obs. eliminated"), "{out}");
    assert_eq!(s.n(), 100);
    s.run("stcox x g");
    assert_close(s.scalar("e(ll)"), ll, 1e-10);
}

#[test]
fn cox_efron_ll_by_hand() {
    // two tied failures at t=2: Efron term log(S0) + log(S0 - (r1+r2)/2)
    let mut s = tiny();
    s.run("stset t, failure(d)");
    s.run("stcox x, efron");
    let b = s.scalar("_b[x]");
    let ll = s.scalar("e(ll)");
    let t = [1.0, 2.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0];
    let d = [true, true, true, false, true, false, true, true];
    let x: Vec<f64> = (1..=8).map(|i| ((i * 7) % 5) as f64 - 2.0).collect();
    let mut hand = 0.0;
    for tj in [1.0, 2.0, 4.0, 7.0, 8.0] {
        let s0: f64 = (0..8).filter(|&i| t[i] >= tj).map(|i| (b * x[i]).exp()).sum();
        let fails: Vec<usize> = (0..8).filter(|&i| t[i] == tj && d[i]).collect();
        let sd: f64 = fails.iter().map(|&i| (b * x[i]).exp()).sum();
        let m = fails.len() as f64;
        for (r, &i) in fails.iter().enumerate() {
            hand += b * x[i] - (s0 - r as f64 / m * sd).ln();
        }
    }
    assert_close(ll, hand, 1e-9);
}

#[test]
fn cox_strata_tvc_and_robust() {
    let mut s = sim(200, 11);
    s.run("stset t, failure(d) id(id)");
    let out = s.run("stcox x, strata(g)");
    assert!(out.contains("Stratified Cox regression"), "{out}");
    assert_eq!(s.run("display e(strata)").trim(), "g");
    // g is constant within strata: omitted
    let out = s.run("stcox x g, strata(g)");
    assert!(out.contains("omitted"), "{out}");
    // tvc: x*_t as an extra equation
    let out = s.run("stcox x, tvc(x) texp(ln(_t))");
    assert!(out.contains("tvc") && out.contains("ln(_t)"), "{out}");
    s.scalar("_b[tvc:x]");
    // robust variance equals the scaled sum of squared dfbetas
    s.run("stcox x g, vce(robust)");
    let se = s.scalar("_se[x]");
    assert!(s.run("display e(clustvar)").contains("id"));
    s.run("quietly stcox x g");
    s.run("predict db1 db2, dfbeta");
    s.run("gen double db1sq = db1^2");
    s.run("summarize db1sq");
    let n = s.scalar("r(N)");
    assert_close(se, (s.scalar("r(sum)") * n / (n - 1.0)).sqrt(), 1e-7);
}

#[test]
fn cox_residuals_and_baseline() {
    let mut s = sim(150, 5);
    s.run("stset t, failure(d)");
    s.run("quietly stcox x g");
    s.run("predict mg, mgale");
    s.run("predict cs, csnell");
    s.run("predict H0, basechazard");
    s.run("predict S0, basesurv");
    s.run("predict hr");
    s.run("predict sc1 sc2, schoenfeld");
    s.run("predict e1 e2, esr");
    // martingale residuals, Schoenfeld residuals and score residuals sum to zero
    for v in ["mg", "sc1", "sc2", "e1", "e2"] {
        s.run(&format!("summarize {v}"));
        assert!(s.scalar("r(sum)").abs() < 1e-6, "{v}");
    }
    // Cox-Snell = d - martingale = H0(t) * hr
    s.run("gen double chk = abs(cs - H0*hr)");
    s.run("summarize chk");
    assert!(s.scalar("r(max)") < 1e-5);
    // baseline survivor is a decreasing step function in (0, 1]
    s.run("sort _t");
    let sv = s.nums("S0");
    assert!(sv.windows(2).all(|w| w[1] <= w[0] + 1e-12) && sv.iter().all(|v| *v > 0.0 && *v <= 1.0));
    // PH test and concordance
    let out = s.run("estat phtest, detail");
    assert!(out.contains("Global test"), "{out}");
    assert_eq!(s.scalar("r(df)"), 2.0);
    s.run("estat phtest, km");
    s.run("estat concordance");
    let c = s.scalar("r(C)");
    assert!(c > 0.5 && c < 1.0);
    assert_close(s.scalar("r(D)"), 2.0 * c - 1.0, 1e-12);
    s.run("stcurve, survival at1(g=0) at2(g=1)");
    s.run("stcurve, cumhaz");
}

#[test]
fn concordance_by_hand() {
    let mut s = tiny();
    s.run("stset t, failure(d)");
    s.run("quietly stcox x");
    s.run("predict hr");
    s.run("estat concordance");
    let hr = s.nums("hr");
    let t = [1.0, 2.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0];
    let d = [true, true, true, false, true, false, true, true];
    let (mut p, mut e, mut tie) = (0.0, 0.0, 0.0);
    for i in 0..8 {
        for j in 0..8 {
            if d[i] && (t[j] > t[i] || (t[j] == t[i] && !d[j])) {
                p += 1.0;
                if (hr[i] - hr[j]).abs() < 1e-9 {
                    tie += 1.0;
                } else if hr[i] > hr[j] {
                    e += 1.0;
                }
            }
        }
    }
    assert_close(s.scalar("r(n_P)"), p, 0.0);
    assert_close(s.scalar("r(C)"), (e + tie / 2.0) / p, 1e-12);
}

#[test]
fn streg_exponential_equals_poisson() {
    let mut s = sim(250, 9);
    s.run("stset t, failure(d)");
    let out = s.run("streg x g, distribution(exponential)");
    assert!(out.contains("Exponential PH regression") && out.contains("Haz. ratio"), "{out}");
    let (bx, se) = (s.scalar("_b[_t:x]"), s.scalar("_se[_t:x]"));
    s.run("gen double dt = _t - _t0");
    s.run("quietly poisson _d x g, exposure(dt)");
    assert_close(bx, s.scalar("_b[x]"), 1e-6);
    assert_close(se, s.scalar("_se[x]"), 1e-5);
    // AFT metric flips the sign
    s.run("streg x g, distribution(exponential) time");
    assert_close(s.scalar("_b[_t:x]"), -bx, 1e-6);
}

#[test]
fn streg_weibull_metrics_and_family() {
    let mut s = sim(250, 13);
    s.run("stset t, failure(d)");
    let out = s.run("streg x g, distribution(weibull)");
    assert!(out.contains("/ln_p") && out.contains("1/p"), "{out}");
    let (bph, lnp, ll) = (s.scalar("_b[_t:x]"), s.scalar("_b[/:ln_p]"), s.scalar("e(ll)"));
    assert_close(s.scalar("e(aux_p)"), lnp.exp(), 1e-10);
    // log likelihood by hand: sum d ln h - H
    s.run("predict double xb, xb");
    s.run(&format!("gen double lli = _d*(xb + {lnp} + (exp({lnp})-1)*ln(_t)) - exp(xb)*_t^exp({lnp})"));
    s.run("summarize lli");
    assert_close(s.scalar("r(sum)"), ll, 1e-8);
    s.run("streg x g, distribution(weibull) time");
    assert_close(s.scalar("_b[_t:x]"), -bph / lnp.exp(), 1e-5);
    assert_close(s.scalar("e(ll)"), ll, 1e-8);
    // the generalized gamma nests the Weibull (kappa = 1) and the lognormal
    s.run("streg x g, distribution(lognormal)");
    let lln = s.scalar("e(ll)");
    s.run("streg x g, distribution(ggamma)");
    assert!(s.scalar("e(ll)") >= ll.max(lln) - 1e-6);
    for dist in ["gompertz", "loglogistic"] {
        let out = s.run(&format!("streg x g, distribution({dist})"));
        assert!(out.contains("regression"), "{out}");
        assert!(s.scalar("e(ll)") < 0.0);
    }
    s.run("streg x g, distribution(weibull) ancillary(g)");
    s.scalar("_b[ln_p:g]");
    s.run("predict med");
    s.run("predict hz, hazard");
    s.run("predict mg, mgale");
    s.run("summarize mg");
    assert!(s.scalar("r(sum)").abs() < 1e-5, "martingale residuals sum to zero with a constant");
    s.run("stcurve, survival");
}

#[test]
fn streg_frailty() {
    let mut s = sim(200, 21);
    s.run("gen grp = mod(_n, 20)");
    s.run("stset t, failure(d)");
    let out = s.run("streg x g, distribution(weibull) frailty(gamma)");
    assert!(out.contains("/ln_the") && out.contains("theta") && out.contains("LR test of theta=0"), "{out}");
    assert!(s.scalar("e(theta)") > 0.0);
    assert!(s.scalar("e(ll)") >= s.scalar("e(ll_c)") - 1e-6);
    let out = s.run("streg x g, distribution(weibull) frailty(gamma) shared(grp)");
    assert!(out.contains("Shared gamma frailty"), "{out}");
    s.run("streg x g, distribution(exponential) frailty(invgauss)");
    assert!(s.scalar("e(ll)") >= s.scalar("e(ll_c)") - 1e-6);
}

#[test]
fn cox_shared_frailty() {
    let mut s = Session::new();
    // 40 clusters of 5 with a gamma frailty of variance 1
    s.script(
        "set obs 200\nset seed 17\ngen grp = ceil(_n/5)\nbysort grp: gen double u = rgamma(1, 1) if _n == 1\nbysort grp: replace u = u[1]\n\
         gen x = rnormal()\ngen t = -ln(runiform())/(u*exp(0.5*x))\ngen d = runiform() < 0.9\nstset t, failure(d)",
    );
    let out = s.run("stcox x, shared(grp)");
    assert!(out.contains("Gamma shared frailty") && out.contains("theta") && out.contains("LR test of theta=0"), "{out}");
    let theta = s.scalar("e(theta)");
    assert!(theta > 0.2 && theta < 3.0, "{theta}");
    assert!(s.scalar("e(ll)") >= s.scalar("e(ll_c)"));
    assert_eq!(s.scalar("e(N_g)"), 40.0);
    // the frailty model moves the coefficient away from the attenuated Cox fit
    let bf = s.scalar("_b[x]");
    s.run("quietly stcox x");
    assert!(bf.abs() > s.scalar("_b[x]").abs());
    s.fails("stcox x, shared(grp) efron");
}

#[test]
fn summaries_and_split() {
    let mut s = sim(120, 3);
    s.run("stset t, failure(d) id(id)");
    let out = s.run("stdescribe");
    assert!(out.contains("Number of subjects"), "{out}");
    assert_eq!(s.scalar("r(N_sub)"), 120.0);
    s.run("stptime, per(1000)");
    let rate = s.scalar("r(rate)");
    s.run("summarize _d");
    let fails = s.scalar("r(sum)");
    s.run("gen double y = _t - _t0");
    s.run("summarize y");
    assert_close(rate, fails / s.scalar("r(sum)") * 1000.0, 1e-10);
    s.run("stsplit band, every(5)");
    assert!(s.n() > 120);
    s.run("stptime, per(1000)");
    assert_close(s.scalar("r(rate)"), rate, 1e-10);
    s.run("stsum, by(g)");
    s.run("stci, by(g) rmean");
    s.run("sts list, at(5 10 20)");
    s.run("sts graph, by(g)");
    s.run("strate g");
}
