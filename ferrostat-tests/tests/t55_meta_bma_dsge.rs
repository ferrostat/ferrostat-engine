//! Wave 7d: meta-analysis, Bayesian model averaging, financial tools and
//! DSGE models. Synthetic data only; checks are closed forms.
//!
//! Syntax of the financial tools (designed for this engine):
//!
//! ```text
//! fin returns pricevar, generate(newvar) [log]
//! fin summarize retvars [, periods(#) rf(#)]
//! fin portfolio retvars [, rf(#) target(#) noshort points(#) graph]
//! fin regress capm retvar mktvar [, rf(var|#)]
//! fin regress fmb retvar factorvars [, lags(#)]       (panel data, xtset)
//! fin valrisk retvar [, method(historical|parametric|montecarlo) level(#)
//!                      horizon(#) value(#) reps(#) rseed(#)]
//! ```
//!
//! Historical VaR is the negated lower (1 - level) percentile of the
//! returns, ES the negated mean of the returns at or below it; parametric
//! VaR is -(mu - z sigma) and ES -(mu - sigma phi(z) / (1 - level)).

use ferrostat_tests::*;

/// Load columns as double variables.
fn data(s: &mut Session, cols: &[(&str, &[f64])]) {
    let n = cols[0].1.len();
    s.run("clear");
    s.run(&format!("set obs {n}"));
    for (name, v) in cols {
        s.run(&format!("gen double {name} = ."));
        for (i, x) in v.iter().enumerate() {
            s.run(&format!("replace {name} = {x:e} in {}", i + 1));
        }
    }
}

const Y: [f64; 6] = [0.10, 0.30, 0.25, 0.60, 0.45, 0.05];
const SE: [f64; 6] = [0.10, 0.15, 0.12, 0.20, 0.18, 0.08];

fn fe(y: &[f64], se: &[f64]) -> (f64, f64, f64) {
    let w: Vec<f64> = se.iter().map(|s| 1.0 / (s * s)).collect();
    let sw: f64 = w.iter().sum();
    let th = y.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>() / sw;
    let q = y.iter().zip(&w).map(|(a, b)| b * (a - th).powi(2)).sum::<f64>();
    (th, 1.0 / sw.sqrt(), q)
}

#[test]
fn meta_fixed_and_dersimonian_laird_closed_forms() {
    let mut s = Session::new();
    data(&mut s, &[("y", &Y), ("se", &SE)]);
    s.run("meta set y se");
    s.run("meta summarize, common");
    let (th, se, q) = fe(&Y, &SE);
    assert_close(s.scalar("r(theta)"), th, 1e-10);
    assert_close(s.scalar("r(se)"), se, 1e-10);
    assert_close(s.scalar("r(Q)"), q, 1e-10);
    assert_close(s.scalar("r(I2)"), 100.0 * (q - 5.0) / q, 1e-8);
    // DerSimonian-Laird
    let w: Vec<f64> = SE.iter().map(|s| 1.0 / (s * s)).collect();
    let sw: f64 = w.iter().sum();
    let sw2: f64 = w.iter().map(|x| x * x).sum();
    let tau2 = ((q - 5.0) / (sw - sw2 / sw)).max(0.0);
    s.run("meta summarize, random(dl)");
    assert_close(s.scalar("r(tau2)"), tau2, 1e-10);
    let ws: Vec<f64> = SE.iter().map(|s| 1.0 / (s * s + tau2)).collect();
    let thr = Y.iter().zip(&ws).map(|(a, b)| a * b).sum::<f64>() / ws.iter().sum::<f64>();
    assert_close(s.scalar("r(theta)"), thr, 1e-10);
    assert_close(s.scalar("r(se)"), 1.0 / ws.iter().sum::<f64>().sqrt(), 1e-10);
    // Hedges estimator: sample variance minus mean within variance
    s.run("meta summarize, random(hedges)");
    let m = Y.iter().sum::<f64>() / 6.0;
    let he = Y.iter().map(|a| (a - m).powi(2)).sum::<f64>() / 5.0 - SE.iter().map(|s| s * s).sum::<f64>() / 6.0;
    assert_close(s.scalar("r(tau2)"), he.max(0.0), 1e-10);
    // REML / ML / EB / PM / SJ run and give non-negative variances
    for m in ["reml", "mle", "ebayes", "sjonkman"] {
        s.run(&format!("meta summarize, random({m})"));
        assert!(s.scalar("r(tau2)") >= 0.0);
    }
    // Paule-Mandel: generalized Q equals k - 1 at the estimate
    s.run("meta summarize, random(pm)");
    let t = s.scalar("r(tau2)");
    let wp: Vec<f64> = SE.iter().map(|s| 1.0 / (s * s + t)).collect();
    let thp = Y.iter().zip(&wp).map(|(a, b)| a * b).sum::<f64>() / wp.iter().sum::<f64>();
    let qg: f64 = Y.iter().zip(&wp).map(|(a, b)| b * (a - thp).powi(2)).sum();
    assert_close(qg, 5.0, 1e-6);
    // prediction interval
    s.run("meta summarize, random(dl) predinterval");
    let half = 0.5 * (s.scalar("r(pi_ub)") - s.scalar("r(pi_lb)"));
    let tcrit = s.scalar("invttail(4, 0.025)");
    assert_close(half, tcrit * (tau2 + 1.0 / ws.iter().sum::<f64>()).sqrt(), 1e-8);
}

#[test]
fn meta_leaveoneout_cumulative_and_subgroups() {
    let mut s = Session::new();
    let order = [3.0, 1.0, 2.0, 6.0, 5.0, 4.0];
    let grp = [1.0, 1.0, 2.0, 2.0, 1.0, 2.0];
    data(&mut s, &[("y", &Y), ("se", &SE), ("x", &order), ("g", &grp)]);
    s.run("meta set y se, common");
    s.run("meta summarize, leaveoneout");
    for i in 0..6 {
        let yy: Vec<f64> = (0..6).filter(|&j| j != i).map(|j| Y[j]).collect();
        let ss: Vec<f64> = (0..6).filter(|&j| j != i).map(|j| SE[j]).collect();
        assert_close(s.scalar(&format!("el(r(leaveoneout), {}, 1)", i + 1)), fe(&yy, &ss).0, 1e-10);
    }
    s.run("meta summarize, cumulative(x)");
    // second row: the two studies with the smallest x (rows 2 and 3)
    assert_close(s.scalar("el(r(cumul), 2, 1)"), fe(&[Y[1], Y[2]], &[SE[1], SE[2]]).0, 1e-10);
    s.run("meta summarize, subgroup(g)");
    let a: Vec<usize> = vec![0, 1, 4];
    assert_close(s.scalar("el(r(esgroup), 1, 1)"), fe(&a.iter().map(|&i| Y[i]).collect::<Vec<_>>(), &a.iter().map(|&i| SE[i]).collect::<Vec<_>>()).0, 1e-10);
    assert!(s.scalar("r(p_Q_b)") > 0.0);
}

#[test]
fn meta_esize_continuous_and_binary() {
    let mut s = Session::new();
    data(
        &mut s,
        &[("n1", &[20.0, 30.0]), ("m1", &[5.0, 6.0]), ("s1", &[2.0, 2.5]), ("n2", &[22.0, 28.0]), ("m2", &[4.5, 5.0]), ("s2", &[2.1, 2.0])],
    );
    let (n1, m1, s1, n2, m2, s2) = (20.0f64, 5.0f64, 2.0f64, 22.0f64, 4.5f64, 2.1f64);
    let sp = (((n1 - 1.0) * s1 * s1 + (n2 - 1.0) * s2 * s2) / (n1 + n2 - 2.0)).sqrt();
    s.run("meta esize n1 m1 s1 n2 m2 s2, esize(mdiff)");
    assert_close(s.scalar("_meta_es[1]"), 0.5, 1e-12);
    assert_close(s.scalar("_meta_se[1]"), (s1 * s1 / n1 + s2 * s2 / n2).sqrt(), 1e-12);
    s.run("meta esize n1 m1 s1 n2 m2 s2, esize(cohend)");
    let d = (m1 - m2) / sp;
    assert_close(s.scalar("_meta_es[1]"), d, 1e-12);
    assert_close(s.scalar("_meta_se[1]"), ((n1 + n2) / (n1 * n2) + d * d / (2.0 * (n1 + n2))).sqrt(), 1e-12);
    s.run("meta esize n1 m1 s1 n2 m2 s2");
    let j = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0);
    assert_close(s.scalar("_meta_es[1]"), j * d, 1e-12);
    s.run("meta esize n1 m1 s1 n2 m2 s2, esize(glassdelta2)");
    assert_close(s.scalar("_meta_es[1]"), (m1 - m2) / s2, 1e-12);
    // binary: a b c d
    data(&mut s, &[("a", &[5.0, 0.0]), ("b", &[15.0, 25.0]), ("c", &[3.0, 2.0]), ("d", &[17.0, 23.0])]);
    s.run("meta esize a b c d");
    assert_close(s.scalar("_meta_es[1]"), (5.0f64 * 17.0 / (15.0 * 3.0)).ln(), 1e-12);
    assert_close(s.scalar("_meta_se[1]"), (1.0f64 / 5.0 + 1.0 / 15.0 + 1.0 / 3.0 + 1.0 / 17.0).sqrt(), 1e-12);
    // zero cell: 0.5 added to every cell of that study
    assert_close(s.scalar("_meta_es[2]"), (0.5f64 * 23.5 / (25.5 * 2.5)).ln(), 1e-12);
    s.run("meta esize a b c d, esize(lnor) zerocells(none)");
    assert_eq!(s.scalar("missing(_meta_es[2])"), 1.0);
    s.run("meta esize a b c d, esize(lnrr)");
    assert_close(s.scalar("_meta_es[1]"), ((5.0f64 / 20.0) / (3.0 / 20.0)).ln(), 1e-12);
    assert_close(s.scalar("_meta_se[1]"), (1.0f64 / 5.0 - 1.0 / 20.0 + 1.0 / 3.0 - 1.0 / 20.0).sqrt(), 1e-12);
    s.run("meta esize a b c d, esize(rdiff)");
    assert_close(s.scalar("_meta_es[1]"), 0.25 - 0.15, 1e-12);
    s.run("meta esize a b c d, esize(lnorpeto)");
    let (nn, e, v) = (40.0f64, 20.0 * 8.0 / 40.0, 20.0 * 20.0 * 8.0 * 32.0 / (1600.0 * 39.0));
    let _ = nn;
    assert_close(s.scalar("_meta_es[1]"), (5.0 - e) / v, 1e-12);
    assert_close(s.scalar("_meta_se[1]"), 1.0 / v.sqrt(), 1e-12);
    // Mantel-Haenszel pooled odds ratio
    s.run("meta update, esize(lnor)");
    s.run("meta summarize, common(mh)");
    let r: f64 = 5.0 * 17.0 / 40.0 + 0.0 * 23.0 / 50.0;
    let q = 15.0 * 3.0 / 40.0 + 25.0 * 2.0 / 50.0;
    assert_close(s.scalar("r(theta)"), (r / q).ln(), 1e-12);
    s.run("meta labbeplot");
    s.run("meta query");
    s.run("meta describe");
}

#[test]
fn meta_bias_tests_and_trimfill() {
    let mut s = Session::new();
    data(&mut s, &[("y", &Y), ("se", &SE)]);
    s.run("meta set y se");
    s.run("meta bias, egger");
    let bias = s.scalar("r(bias)");
    let sb = s.scalar("r(se_bias)");
    s.run("gen double zz = y/se");
    s.run("gen double pr = 1/se");
    s.run("regress zz pr");
    assert_close(bias, s.scalar("_b[_cons]"), 1e-10);
    assert_close(sb, s.scalar("_se[_cons]"), 1e-10);
    s.run("meta bias, begg");
    assert!(s.scalar("r(p)") > 0.0 && s.scalar("r(p)") <= 1.0);
    s.run("meta bias, egger random");
    s.run("meta trimfill");
    assert!(s.scalar("r(K_imputed)") >= 0.0);
    // perfectly symmetric funnel: nothing to impute
    data(&mut s, &[("y", &[0.0, 0.4, 0.1, 0.3, 0.2]), ("se", &[0.2, 0.2, 0.1, 0.1, 0.05])]);
    s.run("meta set y se, common");
    s.run("meta trimfill");
    assert_eq!(s.scalar("r(K_imputed)"), 0.0);
    assert_close(s.scalar("r(theta)"), 0.2, 1e-10);
    s.run("meta trimfill, estimator(run) left funnel");
    s.run("meta forestplot");
    s.run("meta funnelplot");
    s.run("meta funnelplot, contours(1 5 10)");
    s.run("meta galbraithplot");
    // binary tests
    data(&mut s, &[("a", &[5.0, 8.0, 2.0, 12.0, 9.0]), ("b", &[15.0, 22.0, 23.0, 28.0, 26.0]), ("c", &[3.0, 4.0, 2.0, 6.0, 5.0]), ("d", &[17.0, 24.0, 23.0, 34.0, 28.0])]);
    s.run("meta esize a b c d");
    s.run("meta bias, harbord");
    s.run("meta bias, peters");
    assert!(s.scalar("r(p)") > 0.0);
}

#[test]
fn meta_regress_matches_weighted_least_squares() {
    let mut s = Session::new();
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    data(&mut s, &[("y", &Y), ("se", &SE), ("x", &x)]);
    s.run("meta set y se");
    s.run("meta regress x, common");
    let b = s.scalar("_b[x]");
    let c = s.scalar("_b[_cons]");
    s.run("gen double w = 1/se^2");
    s.run("regress y x [aw=w]");
    assert_close(b, s.scalar("_b[x]"), 1e-10);
    assert_close(c, s.scalar("_b[_cons]"), 1e-10);
    // random effects with a constant only reproduces meta summarize
    s.run("meta regress, random(dl)");
    let c0 = s.scalar("_b[_cons]");
    let t0 = s.scalar("e(tau2)");
    s.run("meta summarize, random(dl)");
    assert_close(c0, s.scalar("r(theta)"), 1e-10);
    assert_close(t0, s.scalar("r(tau2)"), 1e-10);
    s.run("meta regress x");
    s.run("estat bubbleplot");
    s.run("predict double xb");
    s.run("predict double fit, fitted");
    assert_close(s.scalar("xb[2]"), s.scalar("_b[_cons] + 2*_b[x]"), 1e-10);
}

#[test]
fn meta_mvregress_and_multilevel() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 30
         set seed 55
         gen id = _n
         gen double v1 = 0.02 + 0.03*runiform()
         gen double v2 = 0.02 + 0.03*runiform()
         gen double v12 = 0
         gen double y1 = 0.3 + 0.2*rnormal()
         gen double y2 = -0.1 + 0.2*rnormal()
         gen double s1 = sqrt(v1)
         gen double s2 = sqrt(v2)",
    );
    // fixed effects with uncorrelated within-study errors: separate IV estimates
    s.run("meta mvregress y1 y2, wcovvariables(v1 v12 v2) fixed");
    let b1 = s.scalar("_b[y1:_cons]");
    let b2 = s.scalar("_b[y2:_cons]");
    s.run("meta set y1 s1");
    s.run("meta summarize, common");
    assert_close(b1, s.scalar("r(theta)"), 1e-10);
    s.run("meta set y2 s2");
    s.run("meta summarize, common");
    assert_close(b2, s.scalar("r(theta)"), 1e-10);
    s.run("meta mvregress y1 y2, wcovvariables(v1 v12 v2)");
    s.run("estat heterogeneity");
    s.run("estat recovariance");
    s.run("estat sd");
    s.run("meta mvregress y1 y2, wcovvariables(v1 v12 v2) random(mle)");
    s.run("meta mvregress y1 y2, wsevariables(s1 s2) random(jwriley)");
    // one level with one study per group = univariate REML random effects
    s.run("meta set y1 s1");
    s.run("meta summarize, random(reml)");
    let tau2 = s.scalar("r(tau2)");
    let th = s.scalar("r(theta)");
    s.run("meta meregress y1 || id:, essevariable(s1)");
    assert_close(s.scalar("el(e(sigma2), 1, 1)"), tau2, 1e-4);
    assert_close(s.scalar("_b[_cons]"), th, 1e-5);
    s.run("estat heterogeneity");
    s.run("estat sd");
    s.run("estat recovariance");
    s.run("gen g = ceil(_n/5)");
    s.run("meta multilevel y1, relevels(g id) essevariable(s1)");
    s.run("estat group");
    assert_eq!(s.scalar("el(r(N_g), 1, 1)"), 6.0);
    s.run("gen nn = id + 10");
    s.run("meta psycorr y1 nn, fisherz");
}

/// Posterior model probabilities under a fixed g by hand: models {},
/// {x1}, {x2}, {x1 x2} with uniform prior.
#[test]
fn bma_enumeration_matches_hand_computed_probabilities() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 60
         set seed 21
         gen double x1 = rnormal()
         gen double x2 = rnormal()
         gen double y = 1 + 0.4*x1 + 0.1*x2 + rnormal()",
    );
    let g = 25.0f64;
    let n = 60.0f64;
    let mut lml = Vec::new();
    for (vars, k) in [("", 0.0), ("x1", 1.0), ("x2", 1.0), ("x1 x2", 2.0)] {
        let r2 = if vars.is_empty() {
            0.0
        } else {
            s.run(&format!("regress y {vars}"));
            s.scalar("e(r2)")
        };
        lml.push(0.5 * (n - 1.0 - k) * (1.0 + g).ln() - 0.5 * (n - 1.0) * (1.0 + g * (1.0 - r2)).ln());
    }
    let mx = lml.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let tot: f64 = lml.iter().map(|v| (v - mx).exp()).sum();
    let p: Vec<f64> = lml.iter().map(|v| (v - mx).exp() / tot).collect();
    s.run("bmaregress y x1 x2, gprior(fixed 25)");
    assert_close(s.scalar("el(e(pip), 1, 1)"), p[1] + p[3], 1e-9);
    let bma_x1 = s.scalar("_b[x1]");
    assert_close(s.scalar("el(e(pip), 1, 2)"), p[2] + p[3], 1e-9);
    // posterior mean of x1: sum over models of PMP * g/(1+g) * OLS coefficient
    s.run("regress y x1");
    let b1 = s.scalar("_b[x1]");
    s.run("regress y x1 x2");
    let b12 = s.scalar("_b[x1]");
    let mean = g / (1.0 + g) * (p[1] * b1 + p[3] * b12);
    assert_close(bma_x1, mean, 1e-9);
    s.run("bmaregress y x1 x2, gprior(fixed 25)");
    s.run("bmastats models");
    s.run("bmastats msize");
    assert_close(s.scalar("r(post_mean)"), p[1] + p[2] + 2.0 * p[3], 1e-9);
    s.run("bmastats pip");
    s.run("bmastats jointness");
    assert_close(s.scalar("el(r(jointness), 1, 1)"), p[3], 1e-9);
    s.run("bmastats lps");
    s.run("bmagraph pmp");
    s.run("bmagraph msize");
    s.run("bmagraph varmap");
    s.run("bmagraph coefdensity x1");
    s.run("bmapredict double yhat");
    s.run("bmacoefsample, rseed(3) mcmcsize(500)");
    // binomial model prior with p = 1/2 equals the uniform prior
    s.run("bmaregress y x1 x2, gprior(fixed 25) mprior(binomial 0.5)");
    assert_close(s.scalar("el(e(pip), 1, 1)"), p[1] + p[3], 1e-9);
    // integrated priors and sampling run
    s.run("bmaregress y x1 x2, gprior(hyper 3)");
    s.run("bmaregress y x1 x2, gprior(zs) mprior(betabinomial 1 1)");
    s.run("bmaregress y x1 x2, gprior(uip) sampling mcmcsize(2000) burnin(200) rseed(9)");
    assert!(s.scalar("el(e(pip), 1, 1)") > 0.5);
}

#[test]
fn fin_capm_portfolio_returns_and_var() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 100
         set seed 8
         gen t = _n
         tsset t
         gen double mkt = 0.01 + 0.04*rnormal()
         gen double a1 = 0.002 + 1.2*mkt + 0.02*rnormal()
         gen double a2 = 0.001 + 0.7*mkt + 0.03*rnormal()
         gen double p = 100*exp(sum(a1))",
    );
    s.run("fin");
    s.run("fin regress capm a1 mkt");
    let beta = s.scalar("e(beta)");
    s.run("correlate a1 mkt, covariance");
    assert_close(beta, s.scalar("r(cov_12)") / s.scalar("r(Var_2)"), 1e-10);
    s.run("fin returns p, generate(lr) log");
    assert_close(s.scalar("lr[5]"), s.scalar("a1[5]"), 1e-9);
    s.run("fin returns p, generate(sr)");
    assert_close(s.scalar("sr[5]"), s.scalar("exp(a1[5]) - 1"), 1e-9);
    // global minimum-variance weights for two assets
    s.run("fin portfolio a1 a2");
    s.run("correlate a1 a2, covariance");
    let (v1, v2, c) = (s.scalar("r(Var_1)"), s.scalar("r(Var_2)"), s.scalar("r(cov_12)"));
    s.run("fin portfolio a1 a2");
    assert_close(s.scalar("el(r(gmv), 1, 1)"), (v2 - c) / (v1 + v2 - 2.0 * c), 1e-9);
    s.run("fin portfolio a1 a2, target(0.01) noshort graph");
    s.run("fin summarize a1 a2, periods(12)");
    s.run("quietly summarize a1");
    let (m, sd) = (s.scalar("r(mean)"), s.scalar("r(sd)"));
    s.run("fin summarize a1");
    assert_close(s.scalar("el(r(stats), 1, 6)"), m / sd, 1e-10);
    s.run("fin valrisk a1, method(parametric) level(95)");
    let z = s.scalar("invnormal(0.95)");
    assert_close(s.scalar("r(VaR)"), -(m - z * sd), 1e-10);
    assert_close(s.scalar("r(ES)"), -(m - sd * s.scalar("normalden(invnormal(0.95))") / 0.05), 1e-10);
    s.run("fin valrisk a1");
    let hv = s.scalar("r(VaR)");
    s.run("_pctile a1, p(5)");
    assert_close(hv, -s.scalar("r(r1)"), 1e-10);
    s.run("fin valrisk a1, method(montecarlo) rseed(1) reps(20000)");
    assert!((s.scalar("r(VaR)") + m - z * sd).abs() < 0.01);
}

#[test]
fn fin_fama_macbeth_averages_cross_sectional_slopes() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 200
         set seed 4
         gen id = ceil(_n/20)
         gen t = mod(_n-1,20)+1
         xtset id t
         gen double beta = id/5 + 0.1*rnormal()
         gen double ret = 0.01 + 0.02*beta + 0.01*rnormal()",
    );
    let mut sum = 0.0;
    for t in 1..=20 {
        s.run(&format!("quietly regress ret beta if t == {t}"));
        sum += s.scalar("_b[beta]");
    }
    s.run("fin regress fmb ret beta");
    assert_close(s.scalar("_b[beta]"), sum / 20.0, 1e-10);
    assert_eq!(s.scalar("e(T)"), 20.0);
    s.run("fin regress fmb ret beta, lags(2)");
}

/// x = beta E x' + z, z' = rho z + e: the policy is x = z / (1 - beta rho).
#[test]
fn dsge_forward_looking_policy_is_analytic() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 250
         set seed 12
         gen t = _n
         tsset t
         gen double z = 0
         replace z = 0.6*z[_n-1] + 0.5*rnormal() in 2/l
         gen double x = z/(1 - 0.5*0.6)
         constraint 1 _b[beta] = 0.5",
    );
    s.run("dsge (x = {beta}*E(F.x) + z) (F.z = {rho}*z, state), constraints(1)");
    let rho = s.scalar("_b[rho]");
    let sd = s.scalar("_b[/sd(e.z)]");
    assert!((rho - 0.6).abs() < 0.15);
    s.run("estat policy");
    assert_close(s.scalar("el(r(policy), 1, 1)"), 1.0 / (1.0 - 0.5 * rho), 1e-8);
    s.run("estat transition");
    assert_close(s.scalar("el(r(transition), 1, 1)"), rho, 1e-8);
    s.run("estat stable");
    assert_eq!(s.scalar("r(stable)"), 1.0);
    s.run("estat covariance");
    let g = 1.0 / (1.0 - 0.5 * rho);
    assert_close(s.scalar("el(r(covariance), 1, 1)"), g * g * sd * sd / (1.0 - rho * rho), 1e-6);
    let dir = temp_dir("t55_dsge_irf");
    s.run(&format!("irf set \"{}/d.irf\", replace", dir.display()));
    s.run("irf create a, step(4)");
    let out = s.run("irf table irf, impulse(z) response(x) noci");
    let line = out.lines().find(|l| l.trim_start_matches('|').trim_start().starts_with("3 ")).expect("step 3 row");
    let v = numbers(line.splitn(3, '|').nth(2).unwrap());
    assert_close(v[0], sd * g * rho.powi(3), 1e-4);
    // the state's own response
    s.run("irf table irf, impulse(z) response(z) noci");
    // nonlinear version: same model in levels with steady state z = 1
    s.run("dsgenl (ln(F.z) = {rho}*ln(z)) (x = {beta}*F.x + z), exostate(z) observed(x) constraints(1)");
    s.run("estat steady");
    assert_close(s.scalar("el(r(steady), 1, 2)"), 2.0, 1e-8);
    s.run("estat policy");
    assert_close(s.scalar("el(r(policy), 1, 1)"), 1.0 / (1.0 - 0.5 * s.scalar("_b[rho]")), 1e-6);
}

/// Three-equation model with a static equation (singular lead matrix):
/// the response of inflation to a monetary shock u with persistence rho_u
/// solves a (1 - beta rho) = kappa b, b (1 - rho) = -(a (psi - rho) + 1).
#[test]
fn dsge_new_keynesian_solution_with_static_equation() {
    let mut s = Session::new();
    s.script(
        "clear
         set obs 50
         set seed 1
         gen t = _n
         tsset t
         gen double p = rnormal()
         gen double r = rnormal()",
    );
    s.run("dsge (p = {beta=0.96}*E(F.p) + {kappa=0.1}*y) (y = E(F.y) - (r - E(F.p) - z), unobserved) (r = {psi=1.5}*p + u) (F.z = {rhoz=0.9}*z, state) (F.u = {rhou=0.5}*u, state), solve");
    s.run("estat policy");
    let (beta, kappa, psi, rho) = (0.96, 0.1, 1.5, 0.5);
    // a (1 - beta rho) = kappa b, b = -(a (psi - rho) + 1) / (1 - rho)
    let a = -kappa / (1.0 - rho) / ((1.0 - beta * rho) + kappa * (psi - rho) / (1.0 - rho));
    assert_close(s.scalar("el(r(policy), 1, 2)"), a, 1e-9);
    s.run("estat stable");
    // indeterminate policy rule (psi < 1) is rejected
    s.fails("dsge (p = {beta=0.96}*E(F.p) + {kappa=0.1}*y) (y = E(F.y) - (r - E(F.p) - z), unobserved) (r = {psi=0.5}*p + u) (F.z = {rhoz=0.9}*z, state) (F.u = {rhou=0.5}*u, state), solve");
}
