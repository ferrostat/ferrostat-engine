//! Survey data (svyset, svydescribe, the svy: prefix with linearized and
//! replicate-weight variances, subpopulations, estat effects / sd) and
//! multiple imputation (mi set / register / impute / passive / xeq /
//! extract / convert / estimate).  Survey variances are checked against
//! hand computations and identities with the robust and cluster-robust
//! estimators; Rubin's rules against hand-computed combinations.

use ferrostat_tests::*;

/// Two strata with two PSUs each: y = 1..8, weights 1 2 1 2 3 1 1 3.
fn strat() -> Session {
    let mut s = Session::new();
    s.script(
        "set obs 8\n\
         gen y = _n\n\
         gen h = 1 + (_n > 4)\n\
         gen u = 1 + mod(floor((_n - 1) / 2), 2)\n\
         gen w = 1\n\
         replace w = 2 in 2\nreplace w = 2 in 4\nreplace w = 3 in 5\nreplace w = 3 in 8\n\
         gen N = 4",
    );
    s
}

/// Simulated clustered data.
fn sim(n: usize) -> Session {
    let mut s = Session::new();
    s.script(&format!(
        "set obs {n}\nset seed 4711\n\
         gen psu = ceil(_n / 5)\ngen st = 1 + mod(psu, 3)\n\
         gen x = rnormal() + psu / 20\ngen w = 1 + runiform() * 3\n\
         gen y = 1 + 0.5 * x + rnormal() + mod(psu, 4) / 3\n\
         gen d = runiform() < invlogit(-0.3 + 0.8 * x)\n\
         gen g = mod(_n, 2)"
    ));
    s
}

#[test]
fn svyset_is_stored_and_survives_save_use() {
    let mut s = strat();
    let out = s.run("svyset u [pw=w], strata(h) fpc(N)");
    assert!(out.contains("Sampling weights: w") && out.contains("Strata 1: h") && out.contains("Sampling unit 1: u") && out.contains("FPC 1: N"), "{out}");
    let dir = temp_dir("svyset_store");
    let f = dir.join("d.dta");
    s.run(&format!("save \"{}\"", f.display()));
    s.run("clear");
    s.run(&format!("use \"{}\"", f.display()));
    let out = s.run("svyset");
    assert!(out.contains("Strata 1: h") && out.contains("FPC 1: N") && out.contains("VCE: linearized"), "{out}");
    s.run("svyset, clear");
    assert!(s.run("svyset").contains("no survey characteristics"));
    s.fails("svy: mean y");
    // multistage and replicate-weight declarations
    s.run("gen j1 = w\ngen j2 = w");
    let out = s.run("svyset u [pw=w], strata(h) fpc(N) || _n");
    assert!(out.contains("Sampling unit 2: <observations>"), "{out}");
    let out = s.run("svyset [pw=w], jkrweight(j1 j2, multiplier(0.5)) vce(jackknife)");
    assert!(out.contains("JK weights") && out.contains("VCE: jackknife"), "{out}");
}

#[test]
fn svydescribe_counts_units() {
    let mut s = strat();
    s.run("svyset u [pw=w], strata(h)");
    let out = s.run("svydescribe");
    assert!(out.contains("Describing stage 1 sampling units"), "{out}");
    assert_eq!(s.scalar("r(N_strata)"), 2.0);
    assert_eq!(s.scalar("r(N_units)"), 4.0);
    assert_eq!(s.scalar("r(N)"), 8.0);
}

#[test]
fn stratified_total_matches_hand_computation() {
    let mut s = strat();
    s.run("svyset u [pw=w], strata(h)");
    let out = s.run("svy: total y");
    assert!(out.contains("Survey: Total estimation") && out.contains("Number of strata = 2") && out.contains("Design df"), "{out}");
    // z = w*y; PSU totals 5, 11 | 21, 31; V = sum_h (z_h1 - z_h2)^2 = 36 + 100
    assert_close(s.scalar("_b[y]"), 68.0, 1e-12);
    assert_close(s.scalar("_se[y]"), 136f64.sqrt(), 1e-10);
    assert_eq!(s.scalar("e(df_r)"), 2.0);
    assert_eq!(s.scalar("e(N_psu)"), 4.0);
    assert_eq!(s.scalar("e(N_pop)"), 14.0);
    // finite population correction: f_h = 2/4
    s.run("svyset u [pw=w], strata(h) fpc(N)");
    s.run("svy: total y");
    assert_close(s.scalar("_se[y]"), 68f64.sqrt(), 1e-10);
    // a sampling rate works like a population count
    s.run("gen f = 0.5");
    s.run("svyset u [pw=w], strata(h) fpc(f)");
    s.run("svy: total y");
    assert_close(s.scalar("_se[y]"), 68f64.sqrt(), 1e-10);
}

#[test]
fn stratified_mean_and_ratio_linearization() {
    let mut s = strat();
    s.run("svyset u [pw=w], strata(h)");
    s.run("svy: mean y");
    let y: Vec<f64> = (1..=8).map(|v| v as f64).collect();
    let w = [1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 1.0, 3.0];
    let sw: f64 = w.iter().sum();
    let m = y.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>() / sw;
    let z: Vec<f64> = (0..8).map(|i| w[i] * (y[i] - m) / sw).collect();
    let t = |a: usize, b: usize| z[a] + z[b];
    let v = (t(0, 1) - t(2, 3)).powi(2) + (t(4, 5) - t(6, 7)).powi(2);
    assert_close(s.scalar("_b[y]"), m, 1e-12);
    assert_close(s.scalar("_se[y]"), v.sqrt(), 1e-10);
    // ratio y/x with x = 2: half the mean of y, SE halved
    s.run("gen x2 = 2");
    s.run("svy: ratio (r: y/x2)");
    assert_close(s.scalar("_b[r]"), m / 2.0, 1e-12);
    assert_close(s.scalar("_se[r]"), v.sqrt() / 2.0, 1e-10);
}

#[test]
fn one_unit_per_row_equals_robust_estimators() {
    let mut s = sim(200);
    // each observation its own PSU, no strata: the linearized variance is
    // the robust (sandwich) variance of the pweighted estimator
    s.run("svyset _n [pw=w]");
    s.run("mean y [pw=w]");
    let se_mean = s.scalar("_se[y]");
    s.run("svy: mean y");
    assert_close(s.scalar("_se[y]"), se_mean, 1e-9);
    assert_eq!(s.scalar("e(df_r)"), 199.0);
    s.run("logit d x [pw=w]");
    let (bl, sel) = (s.scalar("_b[x]"), s.scalar("_se[x]"));
    s.run("svy: logit d x");
    assert_close(s.scalar("_b[x]"), bl, 1e-7);
    assert_close(s.scalar("_se[x]"), sel, 1e-6);
    // without weights the svy mean SE is s/sqrt(n)
    s.run("svyset _n");
    s.run("quietly summarize y");
    let sd = s.scalar("r(sd)");
    s.run("svy: mean y");
    assert_close(s.scalar("_se[y]"), sd / 200f64.sqrt(), 1e-10);
}

#[test]
fn clustered_regress_matches_cluster_robust_up_to_small_sample_factor() {
    let mut s = sim(200);
    s.run("regress y x [pw=w], vce(cluster psu)");
    let (b, se) = (s.scalar("_b[x]"), s.scalar("_se[x]"));
    let (n, k) = (200.0, 2.0);
    s.run("svyset psu [pw=w]");
    let out = s.run("svy: regress y x");
    assert!(out.contains("Survey: Linear regression") && out.contains("Linearized") && out.contains("R-squared"), "{out}");
    assert_close(s.scalar("_b[x]"), b, 1e-9);
    // regress uses G/(G-1) (N-1)/(N-k); the survey estimator only G/(G-1)
    assert_close(s.scalar("_se[x]"), se / ((n - 1.0) / (n - k) as f64).sqrt(), 1e-8);
    assert_eq!(s.scalar("e(df_r)"), 39.0);
    // adjusted Wald F: (d - q + 1)/(d q) W with q = 1 is t^2 (d)/(d)
    let t = s.scalar("_b[x]/_se[x]");
    assert_close(s.scalar("e(F)"), t * t, 1e-8);
    assert_eq!(s.scalar("e(df_m)"), 1.0);
}

#[test]
fn subpop_equals_over_group_and_differs_from_if() {
    let mut s = sim(200);
    s.run("svyset psu [pw=w], strata(st)");
    s.run("svy: mean y, over(g)");
    let b1 = s.scalar("_b[c.y@1.g]");
    let se1 = s.scalar("_se[c.y@1.g]");
    let out = s.run("svy, subpop(g): mean y");
    assert!(out.contains("Subpop. no. obs") && out.contains("Subpop. size"), "{out}");
    assert_close(s.scalar("_b[y]"), b1, 1e-12);
    assert_close(s.scalar("_se[y]"), se1, 1e-10);
    assert_eq!(s.scalar("e(N_sub)"), 100.0);
    assert_eq!(s.scalar("e(N)"), 200.0);
    // subpop(if exp) is the same subpopulation
    s.run("svy, subpop(if g == 1): mean y");
    assert_close(s.scalar("_se[y]"), se1, 1e-10);
    // an if restriction drops the other units from the design instead
    s.run("svy: mean y if g == 1");
    assert_close(s.scalar("_b[y]"), b1, 1e-12);
    assert_eq!(s.scalar("e(N)"), 100.0);
}

#[test]
fn single_unit_strata_treatments() {
    let mut s = strat();
    s.run("set obs 9\nreplace y = 9 in 9\nreplace h = 3 in 9\nreplace u = 1 in 9\nreplace w = 1 in 9");
    s.run("svyset u [pw=w], strata(h)");
    let out = s.run("svy: total y");
    assert!(out.contains("Missing standard errors because of stratum with single sampling unit"), "{out}");
    s.run("svyset u [pw=w], strata(h) singleunit(certainty)");
    s.run("svy: total y");
    assert_close(s.scalar("_se[y]"), 136f64.sqrt(), 1e-10);
    s.run("svyset u [pw=w], strata(h) singleunit(scaled)");
    s.run("svy: total y");
    assert_close(s.scalar("_se[y]"), (136.0 * 3.0 / 2.0f64).sqrt(), 1e-10);
    // centered: the singleton deviates from the mean of all PSU totals
    s.run("svyset u [pw=w], strata(h) singleunit(centered)");
    s.run("svy: total y");
    let totals = [5.0, 11.0, 21.0, 31.0, 9.0];
    let gm = totals.iter().sum::<f64>() / 5.0;
    assert_close(s.scalar("_se[y]"), (136.0 + (9.0f64 - gm).powi(2)).sqrt(), 1e-10);
}

#[test]
fn replicate_weight_variances() {
    let mut s = sim(120);
    // each observation a PSU: the delete-one jackknife of a mean equals s^2/n
    s.run("svyset _n, vce(jackknife)");
    let out = s.run("svy: mean y");
    assert!(out.contains("Jackknife replications (120)") && out.contains("Jackknife *"), "{out}");
    s.run("quietly summarize y");
    let sd = s.scalar("r(sd)");
    s.run("svy: mean y");
    assert_close(s.scalar("_se[y]"), sd / 120f64.sqrt(), 1e-9);
    assert_eq!(s.scalar("e(N_reps)"), 120.0);
    // replicate weights built by hand reproduce the design-based jackknife
    let mut s = strat();
    s.run("svyset u [pw=w], strata(h) vce(jackknife)");
    s.run("svy: total y");
    let se_design = s.scalar("_se[y]");
    s.script(
        "gen j1 = cond(h == 1, cond(u == 1, 0, 2 * w), w)\n\
         gen j2 = cond(h == 1, cond(u == 2, 0, 2 * w), w)\n\
         gen j3 = cond(h == 2, cond(u == 1, 0, 2 * w), w)\n\
         gen j4 = cond(h == 2, cond(u == 2, 0, 2 * w), w)",
    );
    s.run("svyset [pw=w], jkrweight(j1-j4, multiplier(0.5))");
    let out = s.run("svy: total y");
    assert!(out.contains("Replications"), "{out}");
    assert_close(s.scalar("_se[y]"), se_design, 1e-10);
    // for a total with two PSUs per stratum the jackknife equals linearization
    assert_close(se_design, 136f64.sqrt(), 1e-10);
    // BRR with full orthogonal balance reproduces the linearized variance of a total
    s.run("svyset u [pw=w], strata(h) vce(brr)");
    let out = s.run("svy: total y");
    assert!(out.contains("BRR replications"), "{out}");
    assert_close(s.scalar("_se[y]"), 136f64.sqrt(), 1e-10);
    // explicit BRR weights: the half-samples of a 2x2 Hadamard design
    s.script(
        "gen b1 = 2 * w * (u == 1)\n\
         gen b2 = 2 * w * cond(h == 1, u == 2, u == 1)\n\
         gen b3 = 2 * w * cond(h == 1, u == 1, u == 2)\n\
         gen b4 = 2 * w * (u == 2)",
    );
    s.run("svyset [pw=w], brrweight(b1-b4)");
    s.run("svy: total y");
    assert_close(s.scalar("_se[y]"), 136f64.sqrt(), 1e-10);
}

#[test]
fn proportion_and_tabulate() {
    let mut s = sim(300);
    s.run("gen c = 1 + (x > 0) + (x > 1)");
    s.run("svyset _n");
    s.run("proportion c");
    let p = s.scalar("_b[1.c]");
    let out = s.run("svy: proportion c");
    assert!(out.contains("Logit") && out.contains("Proportion"), "{out}");
    assert_close(s.scalar("_b[1.c]"), p, 1e-12);
    // srs: the uncorrected Pearson statistic is the ordinary one
    let tab = s.run("tabulate c d, chi2");
    let chi = stat(&tab, "Pearson chi2(2) =").unwrap();
    let out = s.run("svy: tabulate c d");
    assert!(out.contains("Uncorrected   chi2(2)") && out.contains("Design-based  F("), "{out}");
    assert_close(s.scalar("e(cun_Pear)"), chi, 1e-3);
    // cell proportions add up to one
    let b: f64 = (1..=3).flat_map(|r| (1..=2).map(move |c| (r, c))).map(|(r, c)| s.scalar(&format!("_b[p{r}{c}]"))).sum();
    assert_close(b, 1.0, 1e-12);
    // a Rao-Scott correction close to one under srs
    let f = s.scalar("e(F_Pear)");
    let d1 = s.scalar("e(df1_Pear)");
    assert!(d1 > 1.5 && d1 <= 2.0 + 1e-9, "df1 {d1}");
    assert!((f * d1 / chi - 1.0).abs() < 0.1, "F {f} chi2 {chi}");
    let out = s.run("svy: tabulate c");
    assert!(out.contains("proportion") && out.contains("Total"), "{out}");
}

#[test]
fn models_under_svy() {
    let mut s = sim(300);
    s.run("gen c = 1 + (y > 0.5) + (y > 1.5)");
    s.run("svyset psu [pw=w], strata(st)");
    for cmd in ["svy: probit d x", "svy: poisson c x", "svy: ologit c x", "svy: mlogit c x", "svy: logistic d x"] {
        let out = s.run(cmd);
        assert!(out.contains("Survey:") && out.contains("Linearized") && out.contains("Design df"), "{cmd}\n{out}");
        assert!(s.scalar("e(F)") > 0.0);
    }
    // point estimates are those of the pweighted model
    s.run("ologit c x [pw=w]");
    let b = s.scalar("_b[x]");
    s.run("svy: ologit c x");
    assert_close(s.scalar("_b[x]"), b, 1e-6);
    s.run("mlogit c x [pw=w], baseoutcome(1)");
    let b = s.scalar("_b[2:x]");
    s.run("svy: mlogit c x, baseoutcome(1)");
    assert_close(s.scalar("_b[2:x]"), b, 1e-6);
    // factor variables and postestimation
    s.run("svy: regress y x i.g");
    assert!(s.scalar("_se[1.g]") > 0.0);
    s.run("test x");
}

#[test]
fn estat_effects_and_sd() {
    let mut s = sim(200);
    // srs design without weights: DEFF = DEFT^2 = 1
    s.run("svyset _n");
    s.run("svy: mean y");
    let out = s.run("estat effects");
    assert!(out.contains("DEFF") && out.contains("DEFT"), "{out}");
    let r = row(&out, "y").unwrap();
    assert_close(r[2], 1.0, 1e-4);
    assert_close(r[3], 1.0, 1e-4);
    let out = s.run("estat sd");
    s.run("quietly summarize y");
    let sd = s.scalar("r(sd)");
    assert_close(row(&out, "y").unwrap()[1], sd, 1e-5);
    // clustering inflates the design effect
    s.run("svyset psu");
    s.run("svy: mean y");
    let out = s.run("estat effects");
    assert!(row(&out, "y").unwrap()[2] > 1.0, "{out}");
}

// ---------------------------------------------------------------------------
// multiple imputation
// ---------------------------------------------------------------------------

#[test]
fn rubin_rules_hand_computed() {
    use ferrostat_cmd::matrix::Matrix;
    let bs = vec![vec![1.0], vec![2.0], vec![4.0]];
    let vs = vec![Matrix::from_rows(&[vec![0.5]]), Matrix::from_rows(&[vec![0.7]]), Matrix::from_rows(&[vec![0.6]])];
    let r = ferrostat_ado::mi::rubin(&bs, &vs, None);
    let qbar: f64 = 7.0 / 3.0;
    let ubar: f64 = 0.6;
    let b: f64 = (16.0 / 9.0 + 1.0 / 9.0 + 25.0 / 9.0) / 2.0;
    let t = ubar + (1.0 + 1.0 / 3.0) * b;
    let rvi = (1.0 + 1.0 / 3.0) * b / ubar;
    let nu = 2.0 * (1.0 + 1.0 / rvi).powi(2);
    assert_close(r.b[0], qbar, 1e-12);
    assert_close(r.t[(0, 0)], t, 1e-12);
    assert_close(r.w[(0, 0)], ubar, 1e-12);
    assert_close(r.bm[(0, 0)], b, 1e-12);
    assert_close(r.rvi[0], rvi, 1e-12);
    assert_close(r.df[0], nu, 1e-10);
    assert_close(r.fmi[0], (rvi + 2.0 / (nu + 3.0)) / (rvi + 1.0), 1e-12);
    // Barnard-Rubin small-sample degrees of freedom
    let r = ferrostat_ado::mi::rubin(&bs, &vs, Some(10.0));
    let gamma = (1.0 + 1.0 / 3.0) * b / t;
    let nu_obs = 10.0 * 11.0 / 13.0 * (1.0 - gamma);
    assert_close(r.df[0], 1.0 / (1.0 / nu + 1.0 / nu_obs), 1e-10);
}

/// mi data set by hand: y missing in obs 2 and 5, M = 3 imputations.
fn hand_mi(style: &str) -> Session {
    let mut s = Session::new();
    s.script(&format!(
        "set obs 6\ngen y = _n * 2\nreplace y = . in 2\nreplace y = . in 5\ngen x = _n\n\
         mi set flong\nmi register imputed y\nmi register regular x\nmi set M = 3\n\
         replace y = 3 if _mi_m == 1 & _mi_id == 2\nreplace y = 9 if _mi_m == 1 & _mi_id == 5\n\
         replace y = 5 if _mi_m == 2 & _mi_id == 2\nreplace y = 11 if _mi_m == 2 & _mi_id == 5\n\
         replace y = 4 if _mi_m == 3 & _mi_id == 2\nreplace y = 13 if _mi_m == 3 & _mi_id == 5\n\
         mi convert {style}"
    ));
    s
}

#[test]
fn mi_estimate_combines_by_rubin() {
    for style in ["flong", "mlong", "wide"] {
        let mut s = hand_mi(style);
        let out = s.run("mi estimate: mean y");
        assert!(out.contains("Multiple-imputation estimates") && out.contains("Imputations       =          3"), "{style}\n{out}");
        // per-imputation means and variances of the mean
        let obs = [2.0, 6.0, 8.0, 12.0];
        let imps = [[3.0, 9.0], [5.0, 11.0], [4.0, 13.0]];
        let mut q = Vec::new();
        let mut u = Vec::new();
        for im in imps {
            let v: Vec<f64> = obs.iter().chain(im.iter()).copied().collect();
            let m = v.iter().sum::<f64>() / 6.0;
            let s2 = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / 5.0;
            q.push(m);
            u.push(s2 / 6.0);
        }
        let qbar = q.iter().sum::<f64>() / 3.0;
        let ubar = u.iter().sum::<f64>() / 3.0;
        let b = q.iter().map(|x| (x - qbar).powi(2)).sum::<f64>() / 2.0;
        let t = ubar + (4.0 / 3.0) * b;
        assert_close(s.scalar("_b[y]"), qbar, 1e-12);
        assert_close(s.scalar("_se[y]"), t.sqrt(), 1e-10);
        assert_eq!(s.scalar("e(M_mi)"), 3.0);
        assert_eq!(s.scalar("e(N_mi)"), 6.0);
        // small-sample df with complete-data df 5
        let r = (4.0 / 3.0) * b / ubar;
        let nu_m = 2.0 * (1.0 + 1.0 / r).powi(2);
        let nu_obs = 5.0 * 6.0 / 8.0 * (1.0 - (4.0 / 3.0) * b / t);
        s.run("matrix D = e(df_mi)");
        assert_close(s.scalar("el(D,1,1)"), 1.0 / (1.0 / nu_m + 1.0 / nu_obs), 1e-9);
    }
}

#[test]
fn mi_styles_convert_extract_and_describe() {
    let mut s = hand_mi("mlong");
    // mlong keeps the 6 original rows plus 2 incomplete rows per imputation
    assert_eq!(s.n(), 6 + 3 * 2);
    let out = s.run("mi describe");
    assert!(out.contains("Style:  mlong") && out.contains("imputed:  1; y(2)") && out.contains("regular:  1; x"), "{out}");
    assert_eq!(s.scalar("r(M)"), 3.0);
    assert_eq!(s.scalar("r(N_incomplete)"), 2.0);
    s.run("mi convert wide");
    assert!(s.has_var("_1_y") && s.has_var("_3_y") && !s.has_var("_mi_m"));
    assert_eq!(s.n(), 6);
    s.run("mi convert flong");
    assert_eq!(s.n(), 24);
    s.run("mi extract 2, clear");
    assert_all_close(&s.nums("y"), &[2.0, 5.0, 6.0, 8.0, 11.0, 12.0], 0.0);
    assert!(!s.has_var("_mi_m"));
    let mut s = hand_mi("wide");
    s.run("mi extract 0, clear");
    let y = s.nums("y");
    assert!(y[1].is_nan() && y[4].is_nan());
    s.fails("mi describe");
}

#[test]
fn mi_passive_and_xeq() {
    let mut s = hand_mi("mlong");
    let out = s.run("mi passive: generate ly = 2 * y");
    assert!(out.contains("m=0:") && out.contains("m=3:") && out.contains("(2 missing values generated)"), "{out}");
    s.run("mi extract 3, clear");
    assert_all_close(&s.nums("ly"), &[4.0, 8.0, 12.0, 16.0, 26.0, 24.0], 0.0);
    let mut s = hand_mi("flong");
    let out = s.run("mi xeq 1 2: summarize y");
    assert!(out.contains("m=1 data:") && out.contains("m=2 data:") && out.contains("-> summarize y"), "{out}");
    // mi xeq leaves the mi data as they were
    assert_eq!(s.n(), 24);
}

#[test]
fn mi_impute_methods_fill_only_missing_and_are_reproducible() {
    let mut base = Session::new();
    base.script(
        "set obs 200\nset seed 12345\ngen x = rnormal()\ngen z = rnormal()\n\
         gen y = 1 + x + rnormal()\ngen b = runiform() < invlogit(x)\ngen c = 1 + (x > -0.5) + (x > 0.5)\n\
         replace y = . if runiform() < 0.2\nreplace b = . if runiform() < 0.2\nreplace c = . if runiform() < 0.2\n\
         replace z = . if runiform() < 0.1",
    );
    let setup = "mi set mlong\nmi register imputed y b c z\nmi register regular x";
    let runs = [
        "mi impute regress y x, add(3) rseed(10)",
        "mi impute pmm y x, add(3) rseed(10) knn(5)",
        "mi impute logit b x, add(3) rseed(10)",
        "mi impute ologit c x, add(3) rseed(10)",
        "mi impute mlogit c x, add(3) rseed(10)",
        "mi impute mvn y z = x, add(3) rseed(10) burnin(20) burnbetween(5)",
        "mi impute chained (regress) y (logit) b (ologit) c = x, add(3) rseed(10) burnin(3)",
    ];
    for cmd in runs {
        let mut s = Session::new();
        s.load(base.ds().clone());
        s.script(setup);
        let out = s.run(cmd);
        assert!(out.contains("Observations per m") && out.contains("added =        3"), "{cmd}\n{out}");
        let var = cmd.split_whitespace().nth(3).unwrap().trim_start_matches('(');
        let var = if var == "regress)" || cmd.contains("chained") { "y" } else { var };
        let mut s0 = Session::new();
        s0.load(s.ds().clone());
        s0.run("mi extract 0, clear");
        let orig = s0.nums(var);
        let mut imp = Vec::new();
        for m in 1..=3 {
            let mut sm = Session::new();
            sm.load(s.ds().clone());
            sm.run(&format!("mi extract {m}, clear"));
            let v = sm.nums(var);
            for (a, b) in orig.iter().zip(&v) {
                assert!(!b.is_nan(), "{cmd}: unfilled value");
                if !a.is_nan() {
                    assert_eq!(a, b, "{cmd}: observed value changed");
                }
            }
            imp.push(v);
        }
        if (cmd.contains("pmm") || cmd.contains("logit")) && !cmd.contains("chained") {
            // donors / categories come from the observed values
            let obs: Vec<f64> = orig.iter().copied().filter(|x| !x.is_nan()).collect();
            assert!(imp.iter().flatten().all(|v| obs.contains(v)), "{cmd}");
        }
        // the same seed reproduces the imputations
        let mut s2 = Session::new();
        s2.load(base.ds().clone());
        s2.script(setup);
        s2.run(cmd);
        s2.run("mi extract 2, clear");
        assert_eq!(s2.nums(var).iter().map(|x| x.to_bits()).collect::<Vec<_>>(), imp[1].iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "{cmd}");
    }
    // imputing an unregistered variable is an error; so is a missing add()/replace
    let mut s = Session::new();
    s.load(base.ds().clone());
    s.run("mi set wide");
    s.fails("mi impute regress y x, add(2)");
    s.run("mi register imputed y");
    s.fails("mi impute regress y x");
    s.run("mi impute regress y x, add(2) rseed(1)");
    s.run("mi impute regress y x, replace rseed(2)");
    assert_eq!(s.scalar("r(M_update)"), 2.0);
    let out = s.run("mi estimate: regress y x");
    assert!(out.contains("Within VCE type:") && out.contains("DF adjustment:   Small sample"), "{out}");
}
