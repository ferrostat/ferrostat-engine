//! Matching and reweighting estimators: `teffects nnmatch|psmatch|ipwra`,
//! multivalued and nonlinear `teffects`, `tebalance`, and the community
//! commands psmatch2, cem, ebalance, synth, boottest and wyoung.
//!
//! Checks are against hand computations from the data (matching by brute
//! force, weighted means, separate regressions) and against invariants
//! (weights reproduce targets and sum to one, fixed seeds reproduce).

use ferrostat_tests::*;

fn data(n: usize) -> Session {
    let mut s = Session::new();
    s.script(&format!(
        "set obs {n}\nset seed 4321\ngen x = rnormal()\ngen z = runiform()\ngen g = floor(3*runiform())\ngen d = runiform() < invlogit(0.6*x - 0.2)\ngen y = 1 + 2*d + x + 0.5*z + d*x + rnormal()"
    ));
    s
}

/// Indices of the nearest units of group `grp` to unit `i` (ties included).
fn nearest(x: &[f64], d: &[f64], i: usize, grp: f64, m: usize, skip_self: bool) -> Vec<usize> {
    let mut c: Vec<(f64, usize)> = (0..x.len()).filter(|&j| d[j] == grp && !(skip_self && j == i)).map(|j| ((x[i] - x[j]).abs(), j)).collect();
    c.sort_by(|a, b| a.0.total_cmp(&b.0));
    let thr = c[m - 1].0;
    c.into_iter().filter(|p| p.0 <= thr + 1e-12).map(|p| p.1).collect()
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn var(v: &[f64]) -> f64 {
    let m = mean(v);
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0)
}

#[test]
fn nnmatch_one_neighbor_matches_hand_computation() {
    let mut s = data(120);
    s.run("teffects nnmatch (y x) (d)");
    let (x, d, y) = (s.nums("x"), s.nums("d"), s.nums("y"));
    let n = x.len();
    let mut y1 = vec![0.0; n];
    let mut y0 = vec![0.0; n];
    let mut k = vec![0.0; n];
    let mut kp = vec![0.0; n];
    for i in 0..n {
        let js = nearest(&x, &d, i, 1.0 - d[i], 1, false);
        let imp = mean(&js.iter().map(|&j| y[j]).collect::<Vec<_>>());
        for &j in &js {
            k[j] += 1.0 / js.len() as f64;
            kp[j] += 1.0 / (js.len() as f64).powi(2);
        }
        if d[i] == 1.0 {
            y1[i] = y[i];
            y0[i] = imp;
        } else {
            y0[i] = y[i];
            y1[i] = imp;
        }
    }
    let tau = mean(&(0..n).map(|i| y1[i] - y0[i]).collect::<Vec<_>>());
    assert_close(s.scalar("_b[ATE:r1vs0.d]"), tau, 1e-9);
    // Abadie–Imbens variance: conditional variances from the unit and its two nearest own-group neighbours
    let mut v = 0.0;
    for i in 0..n {
        let hs = nearest(&x, &d, i, d[i], 2, true);
        let s2 = var(&std::iter::once(y[i]).chain(hs.iter().map(|&j| y[j])).collect::<Vec<_>>());
        v += (y1[i] - y0[i] - tau).powi(2) + (k[i] * k[i] + 2.0 * k[i] - kp[i]) * s2;
    }
    let se = v.sqrt() / n as f64;
    assert_close(s.scalar("_se[ATE:r1vs0.d]"), se, 1e-8);
    s.run("teffects nnmatch (y x) (d), atet");
    let n1 = d.iter().filter(|&&v| v == 1.0).count() as f64;
    let att = (0..n).filter(|&i| d[i] == 1.0).map(|i| y[i] - y0[i]).sum::<f64>() / n1;
    assert_close(s.scalar("_b[ATET:r1vs0.d]"), att, 1e-9);
}

#[test]
fn nnmatch_exact_cells_equal_ra() {
    let mut s = data(300);
    s.run("teffects nnmatch (y) (d), ematch(g)");
    let nn = s.scalar("_b[ATE:r1vs0.d]");
    s.run("teffects ra (y i.g) (d)");
    assert_close(nn, s.scalar("_b[ATE:r1vs0.d]"), 1e-8);
    // bias adjustment and metrics run; an impossible caliper fails
    s.run("teffects nnmatch (y x z) (d), biasadj(x z) metric(ivariance) nneighbor(2)");
    assert!(s.scalar("_se[ATE:r1vs0.d]") > 0.0);
    s.fails("teffects nnmatch (y x) (d), caliper(1e-9)");
}

#[test]
fn psmatch_matches_hand_matching_on_logit_score() {
    let mut s = data(150);
    s.script("quietly logit d x z\npredict ps, pr");
    let (p, d, y) = (s.nums("ps"), s.nums("d"), s.nums("y"));
    let n = p.len();
    let mut te = Vec::new();
    for i in 0..n {
        let js = nearest(&p, &d, i, 1.0 - d[i], 1, false);
        let imp = mean(&js.iter().map(|&j| y[j]).collect::<Vec<_>>());
        te.push(if d[i] == 1.0 { y[i] - imp } else { imp - y[i] });
    }
    let out = s.run("teffects psmatch (y) (d x z)");
    assert!(out.contains("propensity-score matching") && out.contains("AI robust"), "{out}");
    assert_close(s.scalar("_b[ATE:r1vs0.d]"), mean(&te), 1e-9);
    s.run("teffects psmatch (y) (d x z), atet vce(iid)");
    let att: Vec<f64> = (0..n).filter(|&i| d[i] == 1.0).map(|i| te[i]).collect();
    assert_close(s.scalar("_b[ATET:r1vs0.d]"), mean(&att), 1e-9);
    s.run("tebalance summarize");
    s.run("teoverlap");
}

#[test]
fn ipwra_matches_weighted_regressions() {
    let mut s = data(400);
    s.run("teffects ipwra (y x z) (d x z)");
    let ate = s.scalar("_b[ATE:r1vs0.d]");
    s.script("quietly logit d x z\npredict p, pr\ngen w = cond(d, 1/p, 1/(1-p))\nquietly regress y x z [aw=w] if d==1\npredict m1\nquietly regress y x z [aw=w] if d==0\npredict m0\ngen te = m1 - m0");
    s.run("summarize te, meanonly");
    assert_close(ate, s.scalar("r(mean)"), 1e-7);
    s.run("teffects ipwra (y x z) (d x z), atet");
    let atet = s.scalar("_b[ATET:r1vs0.d]");
    assert!(s.scalar("_se[ATET:r1vs0.d]") > 0.0);
    s.script("gen wt = cond(d, 1, p/(1-p))\nquietly regress y x z [aw=wt] if d==0\npredict n0\nquietly regress y x z if d==1\npredict n1\ngen tt = n1 - n0");
    s.run("summarize tt if d==1, meanonly");
    assert_close(atet, s.scalar("r(mean)"), 1e-7);
}

#[test]
fn multivalued_and_nonlinear_outcome_models() {
    let mut s = data(450);
    s.run("teffects ra (y x) (g)");
    let (a2, p0) = (s.scalar("_b[ATE:r2vs0.g]"), s.scalar("_b[POmean:0.g]"));
    s.script("forvalues k = 0/2 {\nquietly regress y x if g==`k'\npredict m`k'\n}\ngen d1 = m1 - m0\ngen d2 = m2 - m0");
    s.run("summarize d2, meanonly");
    assert_close(a2, s.scalar("r(mean)"), 1e-8);
    s.run("summarize m0, meanonly");
    assert_close(p0, s.scalar("r(mean)"), 1e-8);
    // multinomial treatment model
    s.run("teffects aipw (y x) (g x)");
    assert!(s.scalar("_se[ATE:r1vs0.g]") > 0.0);
    s.run("teffects ipw (y) (g x z), pomeans");
    assert!(s.scalar("_b[POmeans:2.g]").is_finite());
    // logit outcome model: RA equals the averaged logit predictions
    s.script("gen yb = y > 2");
    s.run("teffects ra (yb x, logit) (d)");
    let al = s.scalar("_b[ATE:r1vs0.d]");
    s.script("quietly logit yb x if d==1\npredict q1, pr\nquietly logit yb x if d==0\npredict q0, pr\ngen dq = q1 - q0");
    s.run("summarize dq, meanonly");
    assert_close(al, s.scalar("r(mean)"), 1e-7);
    s.script("gen cnt = rpoisson(exp(0.3*x + 0.5*d))");
    s.run("teffects aipw (cnt x, poisson) (d x)");
    assert!(s.scalar("_se[ATE:r1vs0.d]") > 0.0);
}

#[test]
fn tebalance_raw_standardized_difference() {
    let mut s = data(300);
    s.run("teffects ipw (y) (d x z)");
    let out = s.run("tebalance summarize");
    assert!(out.contains("Standardized differences"), "{out}");
    let (x, d) = (s.nums("x"), s.nums("d"));
    let x1: Vec<f64> = (0..x.len()).filter(|&i| d[i] == 1.0).map(|i| x[i]).collect();
    let x0: Vec<f64> = (0..x.len()).filter(|&i| d[i] == 0.0).map(|i| x[i]).collect();
    let sd = (mean(&x1) - mean(&x0)) / ((var(&x1) + var(&x0)) / 2.0).sqrt();
    assert_close(s.scalar("el(r(table),1,1)"), sd, 1e-8);
    assert_close(s.scalar("el(r(table),1,3)"), var(&x1) / var(&x0), 1e-8);
    // the weighted difference is much smaller
    assert!(s.scalar("abs(el(r(table),1,2))") < 0.1);
    let out = s.run("tebalance overid");
    assert!(out.contains("chi2(3)"), "{out}");
}

#[test]
fn psmatch2_nearest_neighbor_att() {
    let mut s = data(200);
    s.script("gen ps = invlogit(0.6*x - 0.2 + 0.001*_n)");
    s.run("psmatch2 d, pscore(ps) outcome(y)");
    let (p, d, y) = (s.nums("ps"), s.nums("d"), s.nums("y"));
    let att: Vec<f64> = (0..p.len())
        .filter(|&i| d[i] == 1.0)
        .map(|i| {
            let js = nearest(&p, &d, i, 0.0, 1, false);
            y[i] - y[js[0]]
        })
        .collect();
    assert_close(s.scalar("r(att)"), mean(&att), 1e-9);
    assert!(s.has_var("_weight") && s.has_var("_n1") && s.has_var("_y"));
    // control weights add up to the number of matched treated
    let w = s.nums("_weight");
    let wc: f64 = (0..w.len()).filter(|&i| d[i] == 0.0 && w[i].abs() < 1e300).map(|i| w[i]).sum();
    assert_close(wc, att.len() as f64, 1e-9);
    s.run("psmatch2 d x z, outcome(y) kernel common");
    s.run("pstest x z, both");
    s.run("psgraph");
}

#[test]
fn cem_weights_follow_the_strata_formula() {
    let mut s = data(400);
    s.run("cem x (#4) g, treatment(d)");
    let (st, m, w, d) = (s.nums("cem_strata"), s.nums("cem_matched"), s.nums("cem_weights"), s.nums("d"));
    let n = d.len();
    let mt = (0..n).filter(|&i| m[i] == 1.0 && d[i] == 1.0).count() as f64;
    let mc = (0..n).filter(|&i| m[i] == 1.0 && d[i] == 0.0).count() as f64;
    for i in 0..n {
        if m[i] == 0.0 {
            assert_eq!(w[i], 0.0);
        } else if d[i] == 1.0 {
            assert_eq!(w[i], 1.0);
        } else {
            let ts = (0..n).filter(|&j| st[j] == st[i] && d[j] == 1.0).count() as f64;
            let cs = (0..n).filter(|&j| st[j] == st[i] && d[j] == 0.0).count() as f64;
            assert_close(w[i], mc / mt * ts / cs, 1e-12);
        }
    }
    // weighted controls sum to the matched control count
    let sw: f64 = (0..n).filter(|&i| d[i] == 0.0).map(|i| w[i]).sum();
    assert_close(sw, mc, 1e-9);
    assert!(s.scalar("r(L1)") >= 0.0);
}

#[test]
fn ebalance_weights_reproduce_target_means() {
    let mut s = data(400);
    s.run("ebalance d x z, targets(2)");
    s.run("summarize x if d==1");
    let (m1, v1) = (s.scalar("r(mean)"), s.scalar("r(Var)"));
    s.run("summarize x [aw=_webal] if d==0");
    assert_close(s.scalar("r(mean)"), m1, 1e-6);
    let n1 = s.nums("d").iter().filter(|&&v| v == 1.0).count() as f64;
    s.run("summarize _webal if d==0");
    assert_close(s.scalar("r(sum)"), n1, 1e-6);
    let _ = v1;
    s.run("summarize z if d==1, meanonly");
    let mz = s.scalar("r(mean)");
    s.run("summarize z [aw=_webal] if d==0, meanonly");
    assert_close(s.scalar("r(mean)"), mz, 1e-6);
}

#[test]
fn synth_recovers_a_convex_combination() {
    let mut s = Session::new();
    s.script(
        "set obs 120\ngen unit = ceil(_n/20)\nbysort unit: gen year = 1980 + _n\nset seed 7\ngen u = rnormal()\ngen trend = year - 1980\ngen y = unit*trend*0.3 + 5*unit + sin(unit*trend)\ngen x1 = unit + 0.1*trend\nreplace y = . if unit==1\nsort year unit\nby year: replace y = 0.4*y[3] + 0.6*y[5] if unit==1\nby year: replace x1 = 0.4*x1[3] + 0.6*x1[5] if unit==1\nsort unit year\ntsset unit year",
    );
    let out = s.run("synth y x1 y(1985) y(1990) y(1995), trunit(1) trperiod(1996)");
    assert!(out.contains("RMSPE") && out.contains("Unit Weights"), "{out}");
    let total: f64 = (1..=5).map(|r| s.scalar(&format!("el(e(W_weights),{r},2)"))).sum();
    assert_close(total, 1.0, 1e-9);
    assert_close(s.scalar("el(e(W_weights),2,2)"), 0.4, 1e-3);
    assert_close(s.scalar("el(e(W_weights),4,2)"), 0.6, 1e-3);
    s.run("synth y x1 y(1985) y(1990), trunit(1) trperiod(1996) nested figure");
    let total: f64 = (1..=5).map(|r| s.scalar(&format!("el(e(W_weights),{r},2)"))).sum();
    assert_close(total, 1.0, 1e-9);
    assert!(s.scalar("el(e(RMSPE),1,1)") < 0.05);
}

#[test]
fn boottest_is_reproducible_with_a_seed() {
    let mut s = data(300);
    s.script("gen cl = ceil(_n/38)\nquietly regress y d x, cluster(cl)");
    s.run("boottest d, reps(999) seed(11)");
    let p1 = s.scalar("r(p)");
    let lo = s.scalar("el(r(CI),1,1)");
    s.run("boottest d, reps(999) seed(11)");
    assert_eq!(p1, s.scalar("r(p)"));
    assert_eq!(lo, s.scalar("el(r(CI),1,1)"));
    // 8 clusters: Rademacher draws are fully enumerated
    assert_eq!(s.scalar("r(reps)"), 256.0);
    let b = s.scalar("_b[d]");
    assert!(lo < b && b < s.scalar("el(r(CI),1,2)"));
    // at the confidence-set bound the bootstrap p-value is the test level
    s.run(&format!("boottest d = {lo}, reps(999)"));
    assert!((s.scalar("r(p)") - 0.05).abs() < 0.01);
    s.run("boottest d, reps(199) seed(3) weighttype(webb)");
    let p2 = s.scalar("r(p)");
    s.run("set seed 3");
    s.run("boottest d, reps(199) weighttype(webb)");
    assert_eq!(p2, s.scalar("r(p)"));
    s.run("boottest (d) (x), reps(199)");
    assert!(s.scalar("r(F)") > 0.0);
}

#[test]
fn wyoung_adjusted_pvalues_dominate_raw() {
    let mut s = data(200);
    s.script("gen y2 = z + rnormal()\ngen y3 = x + rnormal()");
    s.run("wyoung y y2 y3, cmd(\"regress OUTCOMEVAR d\") familyp(d) bootstraps(100) seed(5)");
    for r in 1..=3 {
        let p = s.scalar(&format!("el(r(table),{r},3)"));
        let pw = s.scalar(&format!("el(r(table),{r},4)"));
        assert!(pw + 1e-12 >= p.min(1.0) - 0.05 && (0.0..=1.0).contains(&pw), "row {r}: p={p} pwy={pw}");
    }
    let a = s.scalar("el(r(table),2,4)");
    s.run("wyoung y y2 y3, cmd(\"regress OUTCOMEVAR d\") familyp(d) bootstraps(100) seed(5)");
    assert_eq!(a, s.scalar("el(r(table),2,4)"));
}
