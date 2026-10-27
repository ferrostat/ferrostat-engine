//! Resampling prefixes: `bootstrap` (with `vce(bootstrap)` and `estat
//! bootstrap`), `simulate` and `permute`.

use ferrostat_tests::*;

/// 200 observations with heteroskedastic errors and 20 clusters of 10.
fn generated(s: &mut Session) {
    s.script(
        "clear
         set obs 200
         set seed 12345
         gen x = rnormal()
         gen g = ceil(_n/10)
         gen y = 1 + 2*x + rnormal()*(1+abs(x))
         gen d = (y + rnormal() > 1)",
    );
}

/// Text of an e() macro.
fn emac(s: &mut Session, name: &str) -> String {
    s.run(&format!("display \"`e({name})'\"")).trim().to_string()
}

fn sd(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

#[test]
fn bootstrap_regress_posts_results_and_restores_data() {
    let mut s = Session::new();
    generated(&mut s);
    let before = s.nums("y");
    s.run("regress y x");
    let (b, b0) = (s.scalar("_b[x]"), s.scalar("_b[_cons]"));
    let out = s.run("bootstrap, reps(100) seed(1): regress y x");
    assert!(out.contains("(running regress on estimation sample)"));
    assert!(out.contains("Bootstrap replications (100): .........10"));
    assert!(out.contains(" done"));
    assert!(out.contains("Linear regression"));
    assert!(out.contains("Replications  =    100"));
    assert!(out.contains("Wald chi2(1)"));
    assert!(out.contains("|   Observed   Bootstrap"));
    assert!(out.contains("Normal-based"));
    assert!(out.contains("y | coefficient  std. err.       z    P>|z|     [95% conf. interval]"));
    // observed coefficients are the full-sample ones
    assert_close(s.scalar("_b[x]"), b, 1e-12);
    assert_close(s.scalar("_b[_cons]"), b0, 1e-12);
    let row = row(&out, "x").unwrap();
    assert_close(row[0], b, 1e-6);
    assert_close(row[1], s.scalar("_se[x]"), 1e-6);
    assert_close(row[2], b / s.scalar("_se[x]"), 1e-3);
    assert_close(row[4], b - 1.959963984540054 * s.scalar("_se[x]"), 1e-6);
    // e() contents
    assert_eq!(s.scalar("e(N_reps)"), 100.0);
    assert_eq!(s.scalar("e(N)"), 200.0);
    assert_eq!(s.scalar("e(N_misreps)"), 0.0);
    assert_eq!(s.scalar("e(level)"), 95.0);
    assert_eq!(s.scalar("missing(e(df_r))"), 1.0);
    assert_eq!(emac(&mut s, "cmd"), "regress");
    assert_eq!(emac(&mut s, "prefix"), "bootstrap");
    assert_eq!(emac(&mut s, "vcetype"), "Bootstrap");
    assert_close(s.scalar("el(e(se),1,1)"), s.scalar("_se[x]"), 1e-12);
    assert_close(s.scalar("el(e(bias),1,1)"), s.scalar("el(e(b_bs),1,1) - _b[x]"), 1e-12);
    // Wald statistic of the single slope
    assert_close(s.scalar("e(chi2)"), (b / s.scalar("_se[x]")).powi(2), 1e-8);
    // data restored
    assert_eq!(s.n(), 200);
    assert_eq!(s.nums("y"), before);
    // r(table) from the display
    assert_close(s.scalar("el(r(table),2,1)"), s.scalar("_se[x]"), 1e-12);
}

#[test]
fn bootstrap_se_agrees_with_robust_and_analytic_values() {
    let mut s = Session::new();
    generated(&mut s);
    // bootstrap s.e. of the mean ~ sd*sqrt((n-1)/n)/sqrt(n)
    s.run("summarize y");
    let analytic = s.scalar("r(sd)") * (199.0f64 / 200.0).sqrt() / 200f64.sqrt();
    s.run("quietly bootstrap r(mean), reps(2000) seed(11): summarize y");
    assert_close(s.scalar("_se[_bs_1]"), analytic, 0.05);
    // pairs bootstrap of OLS ~ heteroskedasticity-robust standard error
    s.run("regress y x, vce(robust)");
    let robust = s.scalar("_se[x]");
    s.run("quietly bootstrap, reps(1000) seed(12): regress y x");
    assert_close(s.scalar("_se[x]"), robust, 0.1);
}

#[test]
fn bootstrap_named_expressions_and_legend() {
    let mut s = Session::new();
    generated(&mut s);
    s.run("summarize y");
    let (m, sdv) = (s.scalar("r(mean)"), s.scalar("r(sd)"));
    let out = s.run("bootstrap mean=r(mean) sd=r(sd) r(N), reps(50) seed(2): summarize y");
    assert!(out.contains("Bootstrap results"));
    assert!(out.contains("      Command: summarize y"));
    assert!(out.contains("         mean: r(mean)"));
    assert!(out.contains("        _bs_3: r(N)"));
    assert_close(row(&out, "mean").unwrap()[0], m, 1e-6);
    assert_close(s.scalar("_b[sd]"), sdv, 1e-12);
    // r(N) never varies: zero bootstrap variance
    assert_eq!(s.scalar("_se[_bs_3]"), 0.0);
    assert_eq!(emac(&mut s, "cmd"), "bootstrap");
    assert_eq!(emac(&mut s, "exp1"), "r(mean)");
    // a missing exp_list is an error for r-class commands
    s.fails("bootstrap, reps(5): summarize y");
}

#[test]
fn bootstrap_seed_reproducible_and_vce_option() {
    let mut s = Session::new();
    generated(&mut s);
    s.run("quietly bootstrap _b, reps(30) seed(77): regress y x");
    let a = s.scalar("_se[x]");
    s.run("set seed 77");
    s.run("quietly bootstrap, reps(30): regress y x");
    assert_eq!(s.scalar("_se[x]"), a);
    let out = s.run("regress y x, vce(bootstrap, reps(30) seed(77))");
    assert!(out.contains("Bootstrap replications (30)"));
    assert_eq!(s.scalar("_se[x]"), a);
    s.run("quietly regress y x, vce(bs, reps(30) seed(77))");
    assert_eq!(s.scalar("_se[x]"), a);
    // different seed, different replicates
    s.run("quietly bootstrap, reps(30) seed(78): regress y x");
    assert!(s.scalar("_se[x]") != a);
    // logit with vce(bootstrap)
    let out = s.run("logit d x, vce(bootstrap, reps(20) seed(5))");
    assert!(out.contains("Logistic regression"));
    assert!(out.contains("Pseudo R2"));
    assert_eq!(emac(&mut s, "cmd"), "logit");
}

#[test]
fn bootstrap_estimation_sample_strata_size_cluster() {
    let mut s = Session::new();
    generated(&mut s);
    // resampling stays inside e(sample): e(N) is the same in every replicate
    s.run("count if d == 1");
    let n1 = s.scalar("r(N)");
    s.run("quietly bootstrap n=e(N), reps(20) seed(3): regress y x if d == 1");
    assert_eq!(s.scalar("_b[n]"), n1);
    assert_eq!(s.scalar("_se[n]"), 0.0);
    assert_eq!(s.scalar("e(N)"), n1);
    // strata keep the stratum sizes: the mean of the stratum variable is fixed
    s.run("quietly bootstrap r(mean), reps(20) seed(4) strata(d): summarize d");
    assert!(s.scalar("_se[_bs_1]") < 1e-12);
    assert_eq!(s.scalar("e(N_strata)"), 2.0);
    // size() draws that many observations
    s.run("quietly bootstrap r(N), reps(10) seed(5) size(50): summarize y");
    assert_eq!(s.scalar("_b[_bs_1]"), 200.0);
    s.run("matrix B = e(b_bs)");
    assert_eq!(s.scalar("el(B,1,1)"), 50.0);
    // clusters of equal size: always 200 observations; idcluster numbers the draws
    s.script(
        "program define maxid, rclass
            summarize newg, meanonly
            return scalar max = r(max)
            return scalar N = r(N)
         end",
    );
    let out = s.run("bootstrap m=r(max) n=r(N), reps(10) seed(6) cluster(g) idcluster(newg): maxid");
    assert!(out.contains("(Replications based on 20 clusters in g)"));
    assert_eq!(s.scalar("_b[m]"), 20.0);
    assert_eq!(s.scalar("_se[m]"), 0.0);
    assert_eq!(s.scalar("_se[n]"), 0.0);
    assert_eq!(s.scalar("e(N_clust)"), 20.0);
    assert!(!s.has_var("newg"));
}

#[test]
fn bootstrap_saving_and_estat_intervals() {
    let mut s = Session::new();
    generated(&mut s);
    let dir = temp_dir("t15_saving");
    let f = dir.join("bsreps.dta");
    let f = f.to_string_lossy().replace('\\', "/");
    s.run(&format!("quietly bootstrap, reps(99) seed(8) saving(\"{f}\", replace): regress y x"));
    let se = s.scalar("_se[x]");
    let b = s.scalar("_b[x]");
    let (pl, pu) = (s.scalar("el(e(ci_percentile),1,1)"), s.scalar("el(e(ci_percentile),2,1)"));
    let (bl, bu) = (s.scalar("el(e(ci_bc),1,1)"), s.scalar("el(e(ci_bc),2,1)"));
    let z0 = s.scalar("el(e(z0),1,1)");
    let out = s.run("estat bootstrap, all");
    assert!(out.contains("(N)") && out.contains("(P)") && out.contains("(BC)"));
    assert!(out.contains("Key:  N: Normal"));
    let out_bc = s.run("estat bootstrap");
    assert!(out_bc.contains("(BC)") && !out_bc.contains("(P)"));
    s.run(&format!("use \"{f}\", clear"));
    assert_eq!(s.n(), 99);
    assert!(s.has_var("_b_x") && s.has_var("_b_cons"));
    let mut v = s.nums("_b_x");
    // replicates are stored as float
    assert_close(sd(&v), se, 1e-6);
    v.sort_by(|a, b| a.total_cmp(b));
    // 2.5th / 97.5th percentiles of 99 values: observations 3 and 97
    assert_close(pl, v[2], 1e-6);
    assert_close(pu, v[96], 1e-6);
    // bias-corrected: z0 from the share of replicates below the estimate
    let share = v.iter().filter(|&&t| t <= b).count() as f64 / 99.0;
    let z0_manual = s.scalar(&format!("invnormal({share})"));
    assert_close(z0, z0_manual, 1e-5);
    let p1 = s.scalar(&format!("100*normal(2*{z0_manual} - invnormal(0.975))"));
    let k = (99.0 * p1 / 100.0).floor() as usize;
    assert_close(bl, v[k], 1e-6);
    assert!(bu > bl);
}

#[test]
fn bootstrap_bca_and_failed_replications() {
    let mut s = Session::new();
    generated(&mut s);
    s.run("quietly bootstrap r(mean), reps(50) seed(9) bca: summarize y");
    assert!(s.scalar("el(e(ci_bca),1,1)") < s.scalar("el(e(ci_bca),2,1)"));
    let out = s.run("estat bootstrap, bca");
    assert!(out.contains("(BCa)"));
    s.script(
        "program define flaky, rclass
            if runiform() < 0.3 {
                error 498
            }
            summarize y, meanonly
            return scalar m = r(mean)
         end",
    );
    let out = s.run("bootstrap r(m), reps(40) seed(10): flaky");
    assert!(out.contains('x'));
    assert!(s.scalar("e(N_misreps)") > 0.0);
    assert!(out.contains("standard-error estimates include only complete replications"));
    // replay
    let out = s.run("bootstrap");
    assert!(out.contains("Bootstrap results"));
}

#[test]
fn bootstrap_with_prefixes_and_loops() {
    let mut s = Session::new();
    generated(&mut s);
    let out = s.run("quietly bootstrap, reps(10) seed(1): regress y x");
    assert!(out.is_empty());
    s.run("capture bootstrap, reps(5): regress y nosuchvar");
    assert_eq!(s.scalar("_rc"), 111.0);
    assert_eq!(s.n(), 200);
    s.run("capture noisily bootstrap r(mean), reps(5) nodots: summarize y");
    assert_eq!(s.scalar("_rc"), 0.0);
    s.script(
        "matrix R = J(3,1,.)
         forvalues i = 1/3 {
            quietly bootstrap r(mean), reps(10) seed(`i'): summarize x
            matrix R[`i',1] = _se[_bs_1]
         }",
    );
    assert!(s.scalar("el(R,1,1)") > 0.0);
    assert!(s.scalar("el(R,1,1)") != s.scalar("el(R,2,1)"));
    let out = s.run("bs, reps(5) seed(1) nodots notable: regress y x");
    assert!(!out.contains("Bootstrap replications"));
    assert!(!out.contains("Normal-based"));
}

#[test]
fn simulate_rclass_program_replaces_data() {
    let mut s = Session::new();
    s.script(
        "program define onesample, rclass
            syntax [, n(integer 50)]
            clear
            set obs `n'
            gen v = rnormal(5, 2)
            summarize v
            return scalar mean = r(mean)
            return scalar sd = r(sd)
         end",
    );
    s.run("set seed 2024");
    let out = s.run("simulate mean=r(mean) r(sd), reps(400): onesample, n(100)");
    assert!(out.contains("      Command: onesample, n(100)"));
    assert!(out.contains("         mean: r(mean)"));
    assert!(out.contains("       _sim_2: r(sd)"));
    assert!(out.contains("Simulations (400): .........10"));
    assert_eq!(s.n(), 400);
    assert_eq!(s.varnames(), vec!["mean", "_sim_2"]);
    s.run("summarize mean");
    // sampling distribution of the mean: centre 5, spread 2/sqrt(100)
    assert_close(s.scalar("r(mean)"), 5.0, 0.01);
    assert_close(s.scalar("r(sd)"), 0.2, 0.1);
    s.run("summarize _sim_2");
    assert_close(s.scalar("r(mean)"), 2.0, 0.02);
    // same seed, same draws
    let first = s.nums("mean");
    s.run("set seed 2024");
    s.run("quietly simulate mean=r(mean) r(sd), reps(400): onesample, n(100)");
    assert_eq!(s.nums("mean"), first);
    // reps() is required
    s.fails("simulate r(mean): onesample");
}

#[test]
fn simulate_eclass_b_and_se_names() {
    let mut s = Session::new();
    s.script(
        "program define olsdraw, eclass
            clear
            set obs 30
            gen x = rnormal()
            gen y = 1 + 0.5*x + rnormal()
            regress y x
         end",
    );
    s.run("quietly simulate _b _se, reps(200) seed(5): olsdraw");
    assert_eq!(s.n(), 200);
    for v in ["_b_x", "_b_cons", "_se_x", "_se_cons"] {
        assert!(s.has_var(v), "missing {v}");
    }
    s.run("summarize _b_x");
    assert_close(s.scalar("r(mean)"), 0.5, 0.05);
    // default exp_list of an e-class command is _b
    s.run("quietly simulate, reps(20) seed(5): olsdraw");
    assert_eq!(s.varnames(), vec!["_b_x", "_b_cons"]);
    // named _b[] expression and the double option
    s.run("quietly simulate slope=_b[x], reps(5) seed(5) double: olsdraw");
    assert_eq!(s.varnames(), vec!["slope"]);
}

#[test]
fn simulate_inside_loop_with_nodots() {
    let mut s = Session::new();
    s.script(
        "program define coin, rclass
            return scalar h = runiform() < 0.5
         end",
    );
    s.script(
        "matrix M = J(2,1,.)
         forvalues k = 1/2 {
            preserve
            quietly simulate h=r(h), reps(500) seed(`k'): coin
            summarize h, meanonly
            matrix M[`k',1] = r(mean)
            restore
         }",
    );
    assert_close(s.scalar("el(M,1,1)"), 0.5, 0.1);
    let out = s.run("simulate h=r(h), reps(5) nodots: coin");
    assert!(!out.contains("Simulations"));
    assert_eq!(s.n(), 5);
}

#[test]
fn permute_counts_extreme_statistics() {
    let mut s = Session::with_fixture();
    let out = s.run("permute x _b[x], reps(200) seed(3): regress y x");
    assert!(out.contains("Monte Carlo permutation results"));
    assert!(out.contains("Permutation replications (200)"));
    assert!(out.contains("Note: c = #{|T| >= |T(obs)|}"));
    // y is almost linear in x: no permutation comes close
    assert_eq!(s.scalar("el(r(c),1,1)"), 0.0);
    assert_eq!(s.scalar("el(r(p),1,1)"), 0.0);
    assert_eq!(s.scalar("r(N_reps)"), 200.0);
    // exact upper bound of the binomial interval when c = 0
    assert_close(s.scalar("el(r(ci),2,1)"), 1.0 - 0.025f64.powf(1.0 / 200.0), 1e-6);
    let x = s.nums("x");
    assert_eq!(s.nums("x"), x);
    // permuting an unrelated variable: p well above 0
    s.run("quietly permute z r(rho), reps(200) seed(4) right: correlate z cnt");
    let p = s.scalar("el(r(p),1,1)");
    assert!(p > 0.0 && p <= 1.0);
    assert_eq!(s.nums("x"), x);
}
