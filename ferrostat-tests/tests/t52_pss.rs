//! Power, precision and sample-size analysis: `power` methods, `ciwidth`,
//! group-sequential designs (`gsbounds`, `gsdesign`) and user-defined
//! methods. Checked against textbook closed forms and internal consistency
//! (solve for N, then for power, returns at least the requested power).

use ferrostat_tests::*;

fn z(p: f64) -> f64 {
    ferrostat_cmd::dist::invnormal(p)
}

const ZA: f64 = 1.959963984540054; // z_{.975}
const ZB: f64 = 0.8416212335729143; // z_{.80}

// ---------------------------------------------------------------------------
// means
// ---------------------------------------------------------------------------

#[test]
fn onemean_closed_form_and_consistency() {
    let mut s = Session::new();
    // z test: n = ((z_{1-a/2} + z_{1-b}) σ / δ)²
    s.run("power onemean 0 0.5, sd(2) knownsd nfractional");
    assert_close(s.scalar("r(N)"), ((ZA + ZB) * 2.0 / 0.5f64).powi(2), 1e-4);
    // integer solution with the t test: power at N reaches the target, at N-1 it does not
    s.run("power onemean 0 0.5");
    let n = s.scalar("r(N)");
    assert!(s.scalar("r(power)") >= 0.8);
    s.run(&format!("power onemean 0 0.5, n({})", n - 1.0));
    assert!(s.scalar("r(power)") < 0.8);
    // effect size: solving for ma at that N returns power .8
    s.run(&format!("power onemean 0, n({n})"));
    let ma = s.scalar("r(ma)");
    s.run(&format!("power onemean 0 {ma}, n({n})"));
    assert_close(s.scalar("r(power)"), 0.8, 1e-6);
    // one-sided z: n = ((z_{1-a} + z_{1-b}) σ/δ)²
    s.run("power onemean 10 12, sd(4) knownsd onesided nfractional");
    assert_close(s.scalar("r(N)"), ((z(0.95) + ZB) * 4.0 / 2.0f64).powi(2), 1e-4);
}

#[test]
fn twomeans_and_pairedmeans() {
    let mut s = Session::new();
    s.run("power twomeans 0 0.5, knownsds nfractional");
    assert_close(s.scalar("r(N1)"), 2.0 * ((ZA + ZB) / 0.5f64).powi(2), 1e-3);
    assert_close(s.scalar("r(N)"), 4.0 * ((ZA + ZB) / 0.5f64).powi(2), 1e-3);
    // allocation ratio: N1 = (1 + 1/k)(zσ/δ)²
    s.run("power twomeans 0 0.5, knownsds nratio(2) nfractional");
    assert_close(s.scalar("r(N1)"), 1.5 * ((ZA + ZB) / 0.5f64).powi(2), 1e-3);
    assert_close(s.scalar("r(N2)") / s.scalar("r(N1)"), 2.0, 1e-9);
    // t test with integer groups meets the target
    s.run("power twomeans 0 0.5");
    assert_eq!(s.scalar("r(N1)"), 64.0);
    assert!(s.scalar("r(power)") >= 0.8);
    // n1 fixed, compute n2
    s.run("power twomeans 0 0.5, n1(40) compute(N2)");
    assert_eq!(s.scalar("r(N1)"), 40.0);
    let n2 = s.scalar("r(N2)");
    s.run(&format!("power twomeans 0 0.5, n1(40) n2({})", n2 - 1.0));
    assert!(s.scalar("r(power)") < 0.8);
    // paired: sd of differences from sd1, sd2 and corr
    s.run("power pairedmeans 0 0.5, sd1(1) sd2(1) corr(0.5) knownsd nfractional");
    assert_close(s.scalar("r(N)"), ((ZA + ZB) / 0.5f64).powi(2), 1e-4);
    assert_close(s.scalar("r(sd_d)"), 1.0, 1e-12);
}

#[test]
fn cluster_randomized_designs() {
    let mut s = Session::new();
    // design effect 1 + (m-1)ρ inflates the number of clusters
    s.run("power onemean 0 0.3, m(10) rho(0.1) cluster nfractional");
    let n_ind = ((ZA + ZB) / 0.3f64).powi(2);
    assert_close(s.scalar("r(K)") * 10.0, n_ind * (1.0 + 9.0 * 0.1), 1e-3);
    s.run("power twomeans 0 0.3, m1(10) m2(10) rho(0.05)");
    let k1 = s.scalar("r(K1)");
    assert!(s.scalar("r(power)") >= 0.8);
    s.run(&format!("power twomeans 0 0.3, k1({}) k2({}) m1(10) m2(10) rho(0.05)", k1 - 1.0, k1 - 1.0));
    assert!(s.scalar("r(power)") < 0.8);
    // cluster size for a fixed number of clusters
    s.run("power twoproportions 0.3 0.45, k1(15) k2(15) rho(0.01)");
    assert!(s.scalar("r(M1)") >= 1.0);
    s.run("power oneproportion 0.3 0.4, k(40) m(10) rho(0.1)");
    assert!(s.scalar("r(power)") > 0.5);
    s.run("power logrank, hratio(0.6) m1(5) rho(0.1) cluster");
    assert!(s.scalar("r(K1)") > 0.0);
}

// ---------------------------------------------------------------------------
// proportions
// ---------------------------------------------------------------------------

#[test]
fn proportions_closed_forms() {
    let mut s = Session::new();
    // score test: n = (z_a √(p0q0) + z_b √(pa qa))² / (pa − p0)²
    s.run("power oneproportion 0.5 0.65, nfractional");
    let expect = ((ZA * (0.25f64).sqrt() + ZB * (0.65f64 * 0.35).sqrt()) / 0.15).powi(2);
    assert_close(s.scalar("r(N)"), expect, 1e-3);
    // Fleiss formula without continuity correction, per group
    s.run("power twoproportions 0.3 0.5, nfractional");
    let pb: f64 = 0.4;
    let expect = ((ZA * (2.0 * pb * (1.0 - pb)).sqrt() + ZB * (0.3f64 * 0.7 + 0.5 * 0.5).sqrt()) / 0.2).powi(2);
    assert_close(s.scalar("r(N1)"), expect, 1e-3);
    s.run("power twoproportions 0.3 0.5");
    assert_eq!(s.scalar("r(N1)"), 93.0);
    // exact tests need more observations; power stays a probability
    s.run("power twoproportions 0.3 0.5, test(fisher) n(186)");
    let pf = s.scalar("r(power)");
    s.run("power twoproportions 0.3 0.5, n(186)");
    let pc = s.scalar("r(power)");
    assert!(pf < pc && pf > 0.6, "{pf} {pc}");
    s.run("power twoproportions 0.3 0.5, test(lrchi2) n(186)");
    assert_close(s.scalar("r(power)"), pc, 0.03);
    // exact binomial: power at the solution meets the target
    s.run("power oneproportion 0.2 0.4, test(binomial)");
    assert!(s.scalar("r(power)") >= 0.8);
    s.run("power oneproportion 0.2 0.4, test(wald) n(50)");
    assert!(s.scalar("r(power)") > 0.8);
    // alternative through odds ratio / ratio / diff
    s.run("power twoproportions 0.2, oratio(2) n(300)");
    let p_or = s.scalar("r(power)");
    s.run(&format!("power twoproportions 0.2 {}, n(300)", 2.0 * 0.2 / (1.0 - 0.2 + 2.0 * 0.2)));
    assert_close(s.scalar("r(power)"), p_or, 1e-10);
    // McNemar (Connor): n = (z_a √pd + z_b √(pd − d²))² / d²
    s.run("power pairedproportions 0.1 0.25, nfractional");
    let (pd, d) = (0.35f64, 0.15f64);
    assert_close(s.scalar("r(N)"), ((ZA * pd.sqrt() + ZB * (pd - d * d).sqrt()) / d).powi(2), 1e-3);
}

// ---------------------------------------------------------------------------
// variances, correlations, regression
// ---------------------------------------------------------------------------

#[test]
fn variances_correlations_slopes() {
    let mut s = Session::new();
    // chi-squared test of one variance: power at the solution
    s.run("power onevariance 1 2");
    let n = s.scalar("r(N)");
    assert!(s.scalar("r(power)") >= 0.8);
    s.run(&format!("power onevariance 1 2, n({})", n - 1.0));
    assert!(s.scalar("r(power)") < 0.8);
    // sd scale gives the same answer as squared values
    s.run("power onevariance 1 1.5, sd n(30)");
    let a = s.scalar("r(power)");
    s.run("power onevariance 1 2.25, n(30)");
    assert_close(s.scalar("r(power)"), a, 1e-12);
    s.run("power twovariances 1 2, n(100)");
    assert!(s.scalar("r(power)") > 0.5);
    // Fisher's z: n = ((z_a + z_b)/atanh(r))² + 3
    s.run("power onecorrelation 0 0.3, nfractional");
    assert_close(s.scalar("r(N)"), ((ZA + ZB) / (0.3f64).atanh()).powi(2) + 3.0, 1e-3);
    s.run("power twocorrelations 0.1 0.4, nfractional");
    let dz = (0.4f64).atanh() - (0.1f64).atanh();
    assert_close(s.scalar("r(N1)"), 2.0 * ((ZA + ZB) / dz).powi(2) + 3.0, 1e-3);
    // slope test power is a t test with ncp δ·sdx·√N/σ
    s.run("power oneslope 0 0.5, sdx(1) sderror(2) n(128)");
    assert!(s.scalar("r(power)") >= 0.8);
}

#[test]
fn anova_family_matches_t_tests() {
    let mut s = Session::new();
    // two-group one-way ANOVA F test = two-sided two-sample t test
    s.run("power oneway 0 0.5, varerror(1) n(80)");
    let f = s.scalar("r(power)");
    s.run("power twomeans 0 0.5, n(80)");
    assert_close(s.scalar("r(power)"), f, 1e-6);
    // two repeated measures, within factor = paired t test with sd_d² = 2σ²(1−ρ)
    s.run("power repeated 0 0.5, corr(0.5) varerror(1) n(30)");
    let r = s.scalar("r(power)");
    s.run("power pairedmeans 0 0.5, sddiff(1) n(30)");
    assert_close(s.scalar("r(power)"), r, 1e-6);
    // two-way ANOVA: N is a multiple of the number of cells
    s.run("power twoway 1 2 \\ 3 5, varerror(4)");
    assert_eq!(s.scalar("r(N)") % 4.0, 0.0);
    assert!(s.scalar("r(power)") >= 0.8);
    s.run("power twoway 1 2 \\ 3 5, varerror(4) factor(column) n(40)");
    assert!(s.scalar("r(power)") > 0.0);
    s.run("power repeated 1 2 3 \\ 2 3 5, corr(0.5) varerror(4) factor(bwithin) n(100)");
    assert!(s.scalar("r(power)") > 0.0);
    // R² test with one covariate = partial correlation with no controls
    s.run("power rsquared 0.1, n(60)");
    let a = s.scalar("r(power)");
    s.run("power pcorr 0.1, ncontrol(0) n(60)");
    assert_close(s.scalar("r(power)"), a, 1e-10);
    s.run("power rsquared 0.1 0.2, ntested(2) ncontrol(3)");
    assert!(s.scalar("r(power)") >= 0.8);
    s.run("power oneway, ngroups(4) varmeans(0.5) varerror(4) npergroup(20)");
    assert_eq!(s.scalar("r(N)"), 80.0);
}

// ---------------------------------------------------------------------------
// survival, logistic, stratified and matched designs
// ---------------------------------------------------------------------------

#[test]
fn survival_and_logistic_closed_forms() {
    let mut s = Session::new();
    // Schoenfeld for the Cox model: events = (z_a+z_b)² / (σ² b² (1−R²))
    s.run("power cox, hratio(0.5) sd(0.5) eventprob(0.4) nfractional");
    let ev = (ZA + ZB).powi(2) / (0.25 * (0.5f64).ln().powi(2));
    assert_close(s.scalar("r(E)"), ev, 1e-3);
    assert_close(s.scalar("r(N)"), ev / 0.4, 1e-3);
    // log-rank, Schoenfeld: events = 4(z_a+z_b)²/ln(HR)²
    s.run("power logrank, hratio(0.6) schoenfeld nfractional");
    assert_close(s.scalar("r(E)"), 4.0 * (ZA + ZB).powi(2) / (0.6f64).ln().powi(2), 1e-3);
    // Freedman: events = (z_a+z_b)² (1+ψ)²/(1−ψ)²
    s.run("power logrank, hratio(0.6) nfractional");
    assert_close(s.scalar("r(E)"), (ZA + ZB).powi(2) * 1.6f64.powi(2) / 0.4f64.powi(2), 1e-3);
    // survival probabilities give the event probability
    s.run("power logrank 0.5, hratio(0.6) nfractional");
    let pe = 0.5 * 0.5 + 0.5 * (1.0 - 0.5f64.powf(0.6));
    assert_close(s.scalar("r(E)") / s.scalar("r(N)"), pe, 1e-9);
    // exponential, log hazard: N1 = (z_a+z_b)² · 2 / ln(HR)² without censoring
    s.run("power exponential 0.1 0.05, loghazard nfractional");
    assert_close(s.scalar("r(N1)"), (ZA + ZB).powi(2) * 2.0 / (0.5f64).ln().powi(2), 1e-3);
    s.run("power exponential 0.1 0.05, aperiod(2) fperiod(3)");
    assert!(s.scalar("r(E)") < s.scalar("r(N)"));
    // Hsieh, continuous normal covariate: N = (z_a+z_b)² / (p(1−p) b²)
    s.run("power logistic, oratio(1.5) pr(0.3) nfractional");
    assert_close(s.scalar("r(N)"), (ZA + ZB).powi(2) / (0.21 * (1.5f64).ln().powi(2)), 1e-3);
    // general (information-based) method and binary covariate
    s.run("power logistic general, oratio(1.5) pr(0.3) n(250)");
    assert!(s.scalar("r(power)") > 0.7 && s.scalar("r(power)") < 0.9);
    s.run("power logistic, oratio(2) pr(0.3) covprob(0.5)");
    assert!(s.scalar("r(power)") >= 0.8);
    // CMH with one stratum ≈ two-proportion test
    s.run("power cmh 0.2, oratio(2) n(400)");
    let c = s.scalar("r(power)");
    s.run("power twoproportions 0.2, oratio(2) n(400)");
    assert_close(s.scalar("r(power)"), c, 1e-6);
    s.run("power mcc 0.3, oratio(2) m(2)");
    let n = s.scalar("r(N)");
    s.run("power mcc 0.3, oratio(2) m(4)");
    assert!(s.scalar("r(N)") < n, "more controls per case need fewer sets");
    s.run("power trend 0.1 0.2 0.3");
    assert_eq!(s.scalar("r(N)") % 3.0, 0.0);
}

// ---------------------------------------------------------------------------
// tables, graphs, lists
// ---------------------------------------------------------------------------

#[test]
fn lists_tables_and_graphs() {
    let mut s = Session::new();
    let out = s.run("power twomeans 0 (0.3 0.5), n(100 200)");
    assert!(out.contains("N2/N1"));
    assert_eq!(s.scalar("rowsof(r(pss_table))"), 4.0);
    s.run("power onemean 0 0.5, n(10(10)50) table(N power)");
    assert_eq!(s.scalar("rowsof(r(pss_table))"), 5.0);
    s.run("power onemean 0 0.5, alpha(0.01 0.05) power(0.8 0.9) parallel");
    assert_eq!(s.scalar("rowsof(r(pss_table))"), 2.0);
    s.run("power onemean 0 0.5, n(10(10)50) graph(title(\"Power curve\") name(pc, replace))");
    s.run("graph describe pc");
    s.run("power twoproportions 0.3 (0.4 0.5), graph");
    s.fails("power nosuchmethod 1 2");
}

// ---------------------------------------------------------------------------
// precision analysis
// ---------------------------------------------------------------------------

#[test]
fn ciwidth_methods() {
    let mut s = Session::new();
    // known sd: n = (2 z σ / ω)²
    s.run("ciwidth onemean, sd(2) width(1) knownsd nfractional");
    assert_close(s.scalar("r(N)"), (2.0 * ZA * 2.0f64).powi(2), 1e-3);
    // probability of width: solving N for Pr(W ≤ ω) ≥ .9
    s.run("ciwidth onemean, sd(2) width(1) probwidth(0.9)");
    let n = s.scalar("r(N)");
    s.run(&format!("ciwidth onemean, sd(2) width(1) n({n})"));
    assert!(s.scalar("r(Pr_width)") >= 0.9);
    s.run(&format!("ciwidth onemean, sd(2) width(1) n({})", n - 1.0));
    assert!(s.scalar("r(Pr_width)") < 0.9);
    // width at the median of s is close to the width at σ for large n
    s.run("ciwidth onemean, sd(1) n(1000) probwidth(0.5)");
    assert_close(s.scalar("r(width)"), 2.0 * 1.962339 / 1000f64.sqrt(), 2e-4);
    s.run("ciwidth twomeans, sd(2) width(1) knownsd nfractional");
    assert_close(s.scalar("r(N1)"), 2.0 * (2.0 * ZA * 2.0f64).powi(2), 1e-2);
    s.run("ciwidth pairedmeans, sddiff(2) n(30 40 50)");
    assert_eq!(s.scalar("rowsof(r(pss_table))"), 3.0);
    s.run("ciwidth onevariance, variance(4) width(2) probwidth(0.8)");
    assert!(s.scalar("r(Pr_width)") >= 0.8);
    s.run("ciwidth onemean, sd(2) n(20(10)60) graph");
}

// ---------------------------------------------------------------------------
// group-sequential designs
// ---------------------------------------------------------------------------

#[test]
fn group_sequential_constants() {
    let mut s = Session::new();
    // Pocock constant, K = 2, two-sided α = .05
    s.run("gsbounds, nlooks(2) efficacy(pocock)");
    assert_close(s.scalar("el(r(bounds),1,3)"), 2.178, 2e-3);
    assert_close(s.scalar("el(r(bounds),2,3)"), 2.178, 2e-3);
    assert_close(s.scalar("el(r(bounds),2,5)"), 0.05, 1e-5);
    // O'Brien–Fleming, K = 2: c₂ ≈ 1.977, c₁ = c₂√2
    s.run("gsbounds, nlooks(2) efficacy(obfleming)");
    let c2 = s.scalar("el(r(bounds),2,3)");
    assert_close(c2, 1.977, 2e-3);
    assert_close(s.scalar("el(r(bounds),1,3)"), c2 * 2f64.sqrt(), 1e-9);
    // Pocock K = 5 constant ≈ 2.413
    s.run("gsbounds, nlooks(5) efficacy(pocock)");
    assert_close(s.scalar("el(r(bounds),3,3)"), 2.413, 3e-3);
    // a single look is the fixed design
    s.run("gsbounds, nlooks(1) efficacy(obfleming)");
    assert_close(s.scalar("el(r(bounds),1,3)"), ZA, 1e-4);
    assert_close(s.scalar("r(info_ratio)"), 1.0, 1e-4);
    // error spending: cumulative α follows the spending function
    s.run("gsbounds, nlooks(4) efficacy(kdemets(1)) onesided alpha(0.025)");
    for k in 1..=4 {
        assert_close(s.scalar(&format!("el(r(bounds),{k},5)")), 0.025 * k as f64 / 4.0, 1e-5);
    }
    s.run("gsbounds, information(0.3 0.7 1) efficacy(hsdecani(-4)) futility(errpocock) onesided alpha(0.025)");
    assert_close(s.scalar("el(r(bounds),3,6)"), 0.2, 2e-3);
    assert_close(s.scalar("el(r(bounds),3,3)"), s.scalar("el(r(bounds),3,4)"), 1e-9);
    assert!(s.scalar("r(info_ratio)") > 1.0);
    s.run("gsbounds, nlooks(3) efficacy(errobfleming) futility(errobfleming) binding");
    s.run("gsbounds, nlooks(3) efficacy(haybittle)");
    assert_close(s.scalar("el(r(bounds),1,3)"), 3.0, 1e-12);
}

#[test]
fn gsdesign_sample_sizes() {
    let mut s = Session::new();
    s.run("power twomeans 0 0.5, knownsds nfractional");
    let nfix = s.scalar("r(N)");
    s.run("gsdesign twomeans 0 0.5, nlooks(3) efficacy(obfleming)");
    let infl = s.scalar("r(info_ratio)");
    assert_close(s.scalar("r(N_max)"), (nfix * infl - 1e-9).ceil(), 1.0);
    assert_close(s.scalar("r(N_fixed)"), nfix.ceil(), 1e-9);
    s.run("gsdesign onemean 0 0.4, nlooks(4) efficacy(errpocock) futility(errobfleming)");
    assert!(s.scalar("r(N_e1)") < s.scalar("r(N_max)"));
    s.run("gsdesign oneproportion 0.3 0.45, information(0.3 0.6 1) efficacy(kdemets(2))");
    s.run("gsdesign twoproportions 0.3 0.45, nlooks(2) efficacy(pocock)");
    s.run("gsdesign logrank, hratio(0.7) nlooks(3) onesided alpha(0.025)");
    assert!(s.scalar("r(E_max)") > 0.0);
}

// ---------------------------------------------------------------------------
// user-defined methods
// ---------------------------------------------------------------------------

#[test]
fn user_methods() {
    let mut s = Session::new();
    s.script(
        r#"program power_cmd_myz, rclass
               syntax anything, [n(real 10) alpha(real 0.05) power(real -1) sd(real 1) onesided]
               tokenize `anything'
               local d = (`2' - `1')/`sd'
               local p = normal(`d'*sqrt(`n') - invnormal(1-`alpha'/2)) + normal(-`d'*sqrt(`n') - invnormal(1-`alpha'/2))
               return scalar power = `p'
               return scalar delta = `d'
           end
           program ciwidth_cmd_myw, rclass
               syntax , [n(real 10) alpha(real 0.05) sd(real 1) width(real -1)]
               return scalar width = 2*invnormal(1-`alpha'/2)*`sd'/sqrt(`n')
           end"#,
    );
    s.run("power myz 0 0.5, n(40)");
    let p = s.scalar("r(power)");
    s.run("power onemean 0 0.5, n(40) knownsd");
    assert_close(p, s.scalar("r(power)"), 1e-9);
    s.run("power myz 0 0.5");
    let n = s.scalar("r(N)");
    s.run("power onemean 0 0.5, knownsd");
    assert_eq!(n, s.scalar("r(N)"));
    s.run("power myz 0 0.5, n(20(10)50) sd(2) table");
    assert_eq!(s.scalar("rowsof(r(pss_table))"), 4.0);
    s.run("power usermethod myz 0 0.5, n(30)");
    s.run("ciwidth myw, sd(2) width(1)");
    let nw = s.scalar("r(N)");
    s.run("ciwidth onemean, sd(2) width(1) knownsd");
    assert_eq!(nw, s.scalar("r(N)"));
    s.run("gsdesign myz 0 0.5, nlooks(2) efficacy(pocock)");
    assert!(s.scalar("r(N_max)") > 32.0);
}
