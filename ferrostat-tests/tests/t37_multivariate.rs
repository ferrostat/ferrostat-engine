//! Multivariate analysis: `pca`/`pcamat`, `factor`/`factormat`, `rotate`,
//! their `estat` subcommands and `predict`; `cluster` (linkage methods,
//! kmeans, generate, stop, utilities), `clustermat`, `matrix
//! dissimilarity`; `mds`/`mdsmat`, `ca`/`camat`, `mca`, `biplot`;
//! `discrim`, `candisc`; `canon`, `alpha`, `hotelling`, `mvtest`,
//! `procrustes`.
//!
//! Reference values come from two sources: published worked examples that
//! work from a correlation/covariance matrix or a contingency table typed
//! in the session (so no external data are needed), and an independent
//! implementation (numpy/scipy) applied to the deterministic design below.

use ferrostat_tests::*;

/// 60 observations, three groups, four correlated variables (no random numbers).
fn design(s: &mut Session) {
    s.script(
        "clear
         set obs 60
         gen int id = _n
         gen byte g = 1 + mod(_n, 3)
         gen double x1 = mod(_n*_n*17 + 3*_n, 97)/10 + 2*g
         gen double x2 = mod(_n*_n*23 + 7*_n, 89)/10 + 0.5*x1
         gen double x3 = mod(_n*_n*29 + 5*_n, 83)/10 - 0.3*x2
         gen double x4 = mod(_n*_n*13 + 11*_n, 79)/10 + 0.3*x1",
    );
}

/// First `k` elements of row 1 of a matrix expression such as `e(Ev)`.
fn row1(s: &mut Session, m: &str, k: usize) -> Vec<f64> {
    (1..=k).map(|j| s.scalar(&format!("el({m},1,{j})"))).collect()
}

/// Correlations of eight physical measurements of 305 girls.
fn girls(s: &mut Session) {
    s.script(
        "matrix R = (1000,846,805,859,473,398,301,382 \\ 846,1000,881,826,376,326,277,415 \\ 805,881,1000,801,380,319,237,345 \\ 859,826,801,1000,436,329,327,365 \\ 473,376,380,436,1000,762,730,629 \\ 398,326,319,329,762,1000,583,577 \\ 301,277,237,327,730,583,1000,539 \\ 382,415,345,365,629,577,539,1000)
         matrix R = R/1000
         matrix colnames R = height arm_span fore_arm lower_leg weight bitrod ch_girth ch_width
         matrix rownames R = height arm_span fore_arm lower_leg weight bitrod ch_girth ch_width",
    );
}

#[test]
fn pcamat_matches_published_example() {
    let mut s = Session::new();
    s.run("matrix Clow = (41.0707, 37.7273, 57.3232, 28.1313, 44.4444, 119.697, 32.101, 40.8333, 91.2121, 384.775)");
    let out = s.run("pcamat Clow, comp(2) shape(lower) n(100) names(lft500 lft1000 lft2000 lft4000)");
    assert!(out.contains("Principal components/correlation"), "{out}");
    assert!(out.contains("Unexplained"), "{out}");
    let ev = row1(&mut s, "e(Ev)", 4);
    assert_all_close(&ev, &[2.37181, 0.895925, 0.529687, 0.202581], 2e-5);
    assert_close(s.scalar("e(rho)"), 0.8169, 1e-4);
    assert_close(s.scalar("el(e(L),1,1)"), 0.5384, 1e-4);
    assert_close(s.scalar("el(e(L),4,2)"), 0.7770, 1e-4);
    assert_close(s.scalar("el(e(Psi),1,3)"), 0.3387, 1e-4);
    assert_eq!(s.run("display e(cmd)").trim(), "pca");
    assert_eq!(s.scalar("e(f)"), 2.0);
}

#[test]
fn pca_on_data_matches_independent_eigen() {
    let mut s = Session::new();
    design(&mut s);
    s.run("pca x1-x4");
    assert_all_close(&row1(&mut s, "e(Ev)", 4), &[1.78984401, 1.00824596, 0.72698123, 0.47492881], 1e-6);
    // eigenvector signs: positive column sums
    for (i, v) in [0.55643991, 0.58985281, -0.52291999, 0.26268417].iter().enumerate() {
        assert_close(s.scalar(&format!("el(e(L),{},1)", i + 1)), *v, 1e-6);
    }
    // scores have the eigenvalues as variances
    s.run("predict pc1 pc2, score");
    s.run("quietly summarize pc1");
    assert_close(s.scalar("r(Var)"), 1.78984401, 1e-5);
    s.run("estat kmo");
    let k = s.scalar("r(kmo)");
    assert!(k > 0.0 && k < 1.0);
    s.run("estat smc");
    s.run("estat loadings, cnorm(eigen)");
    assert_close(s.scalar("el(r(A),1,1)"), 0.55643991 * 1.78984401f64.sqrt(), 1e-6);
    // vce(normal): standard error of an eigenvalue = lambda*sqrt(2/N)
    s.run("quietly pca x1-x4, comp(2) vce(normal)");
    assert_close(s.scalar("_se[Eigenvalues:Comp1]"), 1.78984401 * (2.0f64 / 60.0).sqrt(), 1e-6);
}

#[test]
fn factor_methods_match_published_values() {
    let mut s = Session::new();
    girls(&mut s);
    let out = s.run("factormat R, n(305) fac(2) ipf");
    assert!(out.contains("iterated principal factors"), "{out}");
    assert!(out.contains("chi2(28) =  2092.68"), "{out}");
    assert_close(s.scalar("el(e(Ev),1,1)"), 4.44901, 2e-4);
    assert_close(s.scalar("el(e(Psi),1,1)"), 0.1620, 2e-4);
    assert_eq!(s.scalar("e(df_m)"), 15.0);
    // varimax: published loadings and rotation matrix
    s.run("rotate");
    assert_close(s.scalar("el(e(r_L),1,1)"), 0.8802, 2e-4);
    assert_close(s.scalar("el(e(r_L),5,2)"), 0.9064, 2e-4);
    assert_close(s.scalar("el(e(r_T),1,1)"), 0.8018, 2e-4);
    assert_close(s.scalar("el(e(r_Ev),1,1)"), 3.39957, 3e-4);
    s.run("rotate, normalize");
    assert_close(s.scalar("el(e(r_L),1,1)"), 0.8724, 2e-4);
    s.run("rotate, quartimax normalize");
    assert_close(s.scalar("el(e(r_T),1,2)"), 0.6188, 2e-4);
    s.run("rotate, oblimin oblique normalize");
    assert_close(s.scalar("el(e(r_L),2,1)"), 0.9560, 2e-4);
    assert_close(s.scalar("el(e(r_Phi),2,1)"), 0.4716, 2e-4);
    assert_close(s.scalar("el(e(r_Ev),1,1)"), 3.95010, 5e-4);
    let c = s.run("estat common");
    assert!(c.contains("oblimin(0) rotated common factors"), "{c}");
    s.run("rotate, promax normalize");
    assert_close(s.scalar("el(e(r_L),1,1)"), 0.8797, 3e-4);
    assert_close(s.scalar("el(e(r_L),5,2)"), 0.9332, 3e-4);
    s.run("matrix W = (1,0 \\ 1,0 \\ 1,0 \\ 1,0 \\ 0,1 \\ 0,1 \\ 0,1 \\ 0,1)");
    s.run("rotate, target(W) normalize");
    assert_close(s.scalar("el(e(r_T),1,1)"), 0.7817, 2e-4);
    s.run("rotate, clear");
    s.fails("estat rotatecompare");
}

#[test]
fn factor_pf_and_ml_on_data() {
    let mut s = Session::new();
    design(&mut s);
    s.run("factor x1-x4");
    assert_all_close(&row1(&mut s, "e(Ev)", 4), &[1.02515146, 0.12872557, -0.07869536, -0.28780724], 1e-6);
    assert_eq!(s.scalar("e(f)"), 2.0);
    assert_eq!(s.run("display e(method)").trim(), "pf");
    // pcf retains eigenvalues >= 1 of R
    s.run("quietly factor x1-x4, pcf");
    assert_eq!(s.scalar("e(f)"), 2.0);
    assert_close(s.scalar("el(e(Ev),1,1)"), 1.78984401, 1e-6);
    // ml: log likelihood and LR test are consistent with the fitted uniquenesses
    s.run("quietly factor x1-x4, ml factors(1)");
    let ll = s.scalar("e(ll)");
    assert!(ll < 0.0);
    assert_close(s.scalar("e(aic)"), -2.0 * ll + 2.0 * 4.0, 1e-8);
    // regression scoring
    let out = s.run("predict f1");
    assert!(out.contains("regression scoring"), "{out}");
    let out = s.run("predict fb, bartlett");
    assert!(out.contains("Bartlett"), "{out}");
}

#[test]
fn cluster_linkage_groups_match_independent_implementation() {
    let mut s = Session::new();
    design(&mut s);
    let sizes = |s: &mut Session, var: &str| -> Vec<f64> {
        let mut c: Vec<f64> = Vec::new();
        let v = s.nums(var);
        let k = v.iter().cloned().fold(0.0, f64::max) as usize;
        for g in 1..=k {
            c.push(v.iter().filter(|&&x| x == g as f64).count() as f64);
        }
        c.sort_by(|a, b| a.total_cmp(b));
        c
    };
    s.run("cluster averagelinkage x1-x4, name(avg)");
    s.run("cluster generate ga = groups(2/4)");
    assert_eq!(sizes(&mut s, "ga2"), vec![26.0, 34.0]);
    assert_eq!(sizes(&mut s, "ga3"), vec![5.0, 26.0, 29.0]);
    assert_eq!(sizes(&mut s, "ga4"), vec![5.0, 12.0, 14.0, 29.0]);
    s.run("cluster completelinkage x1-x4, name(comp)");
    s.run("cluster generate gc = groups(3)");
    assert_eq!(sizes(&mut s, "gc"), vec![8.0, 23.0, 29.0]);
    s.run("cluster wardslinkage x1-x4, name(wrd)");
    s.run("cluster generate gw = groups(4)");
    assert_eq!(sizes(&mut s, "gw"), vec![12.0, 14.0, 16.0, 18.0]);
    s.run("cluster singlelinkage x1-x4, name(sng)");
    s.run("cluster generate gs = groups(2)");
    assert_eq!(sizes(&mut s, "gs"), vec![2.0, 58.0]);
    assert!(s.has_var("sng_id") && s.has_var("sng_ord") && s.has_var("sng_hgt"));
    let l = s.run("cluster list sng");
    assert!(l.contains("type: hierarchical,  method: single,  dissimilarity: L2"), "{l}");
    let d = s.run("cluster dir");
    assert_eq!(d.lines().next().unwrap().trim(), "sng");
    s.run("cluster rename sng single1");
    s.run("cluster drop single1");
    assert!(!s.run("cluster dir").contains("single1"));
}

#[test]
fn cluster_kmeans_and_stop_rule() {
    let mut s = Session::new();
    design(&mut s);
    s.run("cluster kmeans x1-x4, k(3) start(firstk) name(km)");
    let g = s.nums("km");
    let counts: Vec<usize> = (1..=3).map(|k| g.iter().filter(|&&x| x == k as f64).count()).collect();
    assert_eq!(counts, vec![19, 15, 26]);
    let out = s.run("cluster stop km");
    assert!(out.contains("pseudo-F"), "{out}");
    assert_close(s.scalar("r(calinski_3)"), 22.172028226026207, 1e-6);
    // Duda-Hart for one group equals Calinski-Harabasz for two
    s.run("cluster averagelinkage x1-x4, name(av)");
    s.run("cluster stop av, rule(calinski) groups(2)");
    let ch2 = s.scalar("r(calinski_2)");
    s.run("cluster stop av, rule(duda) groups(1)");
    assert_close(s.scalar("r(dudat2_1)"), ch2, 1e-8);
}

#[test]
fn matrix_dissimilarity_and_clustermat() {
    let mut s = Session::new();
    s.script(
        "clear
         input a b c
         1 2 0
         3 1 4
         0 0 1
         end",
    );
    s.run("matrix dissimilarity D = a b c, canberra");
    // |1-3|/4 + |2-1|/3 + |0-4|/4
    assert_close(s.scalar("el(D,2,1)"), 0.5 + 1.0 / 3.0 + 1.0, 1e-12);
    s.run("matrix dissimilarity E = a b c, L1");
    assert_close(s.scalar("el(E,3,2)"), 3.0 + 1.0 + 3.0, 1e-12);
    s.run("matrix dissimilarity M = a b c, Linfinity");
    assert_close(s.scalar("el(M,3,1)"), 2.0, 1e-12);
    // binary measures between variables
    s.script(
        "clear
         input v1 v2 v3
         1 1 0
         1 0 0
         0 1 1
         1 1 1
         end",
    );
    s.run("matrix dissimilarity J = v1 v2 v3, variables Jaccard");
    // v1, v2: a=2, b=1, c=1 -> 0.5
    assert_close(s.scalar("el(J,2,1)"), 0.5, 1e-12);
    s.run("matrix dissimilarity S = v1 v2 v3, variables matching dissim(oneminus)");
    assert_close(s.scalar("el(S,2,1)"), 0.5, 1e-12);
    // clustermat on a small distance matrix: points 0, 1, 10, 11 on a line
    s.run("matrix P = (0,1,10,11 \\ 1,0,9,10 \\ 10,9,0,1 \\ 11,10,1,0)");
    s.run("clustermat singlelinkage P, clear name(pm)");
    assert_eq!(s.n(), 4);
    s.run("cluster generate two = groups(2)");
    assert_eq!(s.nums("two"), vec![1.0, 1.0, 2.0, 2.0]);
    s.run("cluster generate cut5 = cut(5)");
    assert_eq!(s.nums("cut5"), vec![1.0, 1.0, 2.0, 2.0]);
}

#[test]
fn mds_classical_and_modern() {
    let mut s = Session::new();
    design(&mut s);
    let out = s.run("mds x1 x2, id(id) noplot");
    assert!(out.contains("Classical metric multidimensional scaling"), "{out}");
    assert_close(s.scalar("el(e(Ev),1,1)"), 703.202542, 1e-5);
    assert_close(s.scalar("el(e(Ev),1,2)"), 285.049124, 1e-5);
    assert_close(s.scalar("e(mardia1)"), 1.0, 1e-9);
    // two variables in two dimensions: configuration distances reproduce the data
    s.run("matrix Y = e(Y)");
    let d12 = ((s.scalar("el(Y,1,1)") - s.scalar("el(Y,2,1)")).powi(2) + (s.scalar("el(Y,1,2)") - s.scalar("el(Y,2,2)")).powi(2)).sqrt();
    let x1 = s.nums("x1");
    let x2 = s.nums("x2");
    assert_close(d12, ((x1[0] - x1[1]).powi(2) + (x2[0] - x2[1]).powi(2)).sqrt(), 1e-8);
    // modern scaling of exactly Euclidean data has zero stress
    s.run("quietly mds x1 x2, id(id) method(modern) noplot nolog");
    assert!(s.scalar("e(critval)") < 1e-4);
    s.run("matrix P = (0,3,4 \\ 3,0,5 \\ 4,5,0)");
    s.run("mdsmat P, noplot");
    assert_eq!(s.scalar("e(N)"), 3.0);
    assert_close(s.scalar("e(mardia2)"), 1.0, 1e-9);
}

#[test]
fn ca_from_published_table() {
    let mut s = Session::new();
    s.run("matrix F = (4,2,3,2 \\ 4,3,7,4 \\ 25,10,12,4 \\ 18,24,33,13 \\ 10,6,7,2)");
    s.run("matrix rownames F = Senior_mngr Junior_mngr Senior_empl Junior_empl Secretary");
    s.run("matrix colnames F = None Light Medium Heavy");
    let out = s.run("camat F");
    assert!(out.contains("Correspondence analysis"), "{out}");
    assert_close(s.scalar("e(chi2)"), 16.44, 5e-3);
    assert_close(s.scalar("e(inertia)"), 0.0851899, 1e-6);
    assert_close(s.scalar("el(e(Sv),1,1)"), 0.2734211, 1e-6);
    assert_close(s.scalar("el(e(Sv),1,2)"), 0.1000859, 1e-6);
    // symmetric normalization coordinates and statistics
    assert_close(s.scalar("el(e(TR),3,1)"), 0.728, 1e-3);
    assert_close(s.scalar("el(e(TC),1,1)"), 0.752, 1e-3);
    assert_close(s.scalar("el(e(TR),2,2)"), 0.769, 1e-3);
    assert_close(s.scalar("el(e(GSR),2,2)"), 0.991, 1e-3);
}

#[test]
fn discrim_lda_qda_knn_logistic() {
    let mut s = Session::new();
    design(&mut s);
    s.run("discrim lda x1-x4, group(g) notable");
    // resubstitution: number correctly classified (independent computation: 29)
    let correct = s.scalar("el(e(ctable),1,1) + el(e(ctable),2,2) + el(e(ctable),3,3)");
    assert_eq!(correct, 29.0);
    s.run("estat errorrate");
    s.run("estat loadings, unstandardized");
    s.run("estat structure");
    let out = s.run("estat canontest");
    assert!(out.contains("Canon."), "{out}");
    s.run("predict cls");
    s.run("predict double p1 p2 p3, pr");
    s.run("gen double ps = p1 + p2 + p3");
    for v in s.nums("ps") {
        assert_close(v, 1.0, 1e-10);
    }
    s.run("quietly discrim qda x1-x4, group(g)");
    s.run("quietly discrim knn x1-x4, group(g) k(3)");
    s.run("quietly discrim logistic x1-x4, group(g)");
    let out = s.run("candisc x1-x4, group(g)");
    assert!(out.contains("Canonical linear discriminant analysis"), "{out}");
}

#[test]
fn canon_alpha_hotelling() {
    let mut s = Session::new();
    design(&mut s);
    s.run("canon (x1 x2) (x3 x4)");
    assert_close(s.scalar("el(e(ccorr),1,1)"), 0.43122775, 1e-6);
    assert_close(s.scalar("el(e(ccorr),1,2)"), 0.23964388, 1e-6);
    s.run("alpha x1-x4, std");
    // x3 enters reversed (negative loading on the first component)
    assert_close(s.scalar("r(alpha)"), 0.5642886245615227, 1e-8);
    assert_close(s.scalar("r(rho)"), 0.24458423033372143, 1e-8);
    s.run("alpha x1-x4, std asis item");
    assert_close(s.scalar("r(alpha)"), -0.05324260175523889, 1e-8);
    s.run("gen byte one = g == 1");
    s.run("hotelling x1 x2, by(one)");
    assert_close(s.scalar("r(T2)"), 13.805803431234654, 1e-8);
}

#[test]
fn mvtest_and_procrustes() {
    let mut s = Session::new();
    design(&mut s);
    s.run("mvtest normality x1-x4, stats(all)");
    assert_close(s.scalar("r(mskew)"), 0.6549748773191613, 1e-8);
    assert_close(s.scalar("r(mkurt)"), 18.822079487157712, 1e-8);
    // equality of means across groups: four standard statistics
    let out = s.run("mvtest means x1-x4, by(g)");
    assert!(out.contains("Wilks' lambda"), "{out}");
    // one-sample: Hotelling T2 that means equal a vector equals hotelling of deviations
    s.run("mvtest means x1 x2, zero");
    let t2 = s.scalar("r(T2)");
    s.run("hotelling x1 x2");
    assert_close(t2, s.scalar("r(T2)"), 1e-8);
    s.run("mvtest covariances x1-x4, diagonal");
    assert!(s.scalar("r(chi2)") > 0.0);
    s.run("mvtest correlations x1-x4, compound");
    s.run("procrustes (x1 x2) (x3 x4)");
    assert_close(s.scalar("e(rho)"), 0.41469306994002736, 1e-8);
    assert_close(s.scalar("e(P)"), 0.873731989008452, 1e-8);
}
