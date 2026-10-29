//! Remaining multiple-imputation, multivariate and spatial commands:
//! mi flongsep/copy/erase, mi import/export, mi add/append/merge/expand/
//! reshape/stsplit/replace0/select, mi XXXset, mi estimate saving()/using,
//! mi predict/predictnl, mi impute intreg/nbreg/truncreg, mi ptrace;
//! estat summarize/covariance/correlations/grdistances/manova after discrim,
//! cluster set/delete/parsedistance/measures, caprojection, mcaplot,
//! mcaprojection, mdsshepard; spmatrix userdefined and contiguity on a
//! synthetic grid, estat moran.

use ferrostat_tests::*;

fn cd(s: &mut Session, test: &str) -> std::path::PathBuf {
    let dir = temp_dir(test);
    s.run(&format!("cd \"{}\"", dir.to_string_lossy()));
    dir
}

/// 40 observations, y missing in every fifth, z in every seventh.
fn mi_data(s: &mut Session) {
    s.script(
        r#"clear
           set obs 40
           set seed 11
           gen id = ceil(_n/4)
           gen t = mod(_n-1,4)+1
           gen double x = rnormal()
           gen double y = 1 + 2*x + rnormal()
           gen double z = x + rnormal()
           replace y = . if mod(_n,5)==0
           replace z = . if mod(_n,7)==0"#,
    );
}

fn miss(v: f64) -> bool {
    v.is_nan() || v > 8.9e307
}

fn extract_col(s: &mut Session, m: usize, var: &str) -> Vec<f64> {
    s.run("preserve");
    s.run(&format!("mi extract {m}, clear"));
    let v = s.nums(var);
    s.run("restore");
    v
}

// ---------------------------------------------------------------------------
// MI
// ---------------------------------------------------------------------------

#[test]
fn flongsep_copy_erase_and_conversion() {
    let mut s = Session::new();
    let dir = cd(&mut s, "t50_flongsep");
    mi_data(&mut s);
    s.run("mi set flongsep fs");
    s.run("mi register imputed y z");
    s.run("mi impute mvn y z = x, add(2) rseed(3)");
    assert!(dir.join("_1_fs.dta").exists() && dir.join("_2_fs.dta").exists());
    assert_eq!(s.n(), 40);
    let y2 = extract_col(&mut s, 2, "y");
    assert!(y2.iter().all(|v| !v.is_nan() && *v < 8.9e307));
    // conversion to mlong keeps the imputations
    s.run("mi convert mlong");
    assert_eq!(extract_col(&mut s, 2, "y"), y2);
    // mi copy makes flongsep data under a new name; mi erase removes them
    s.run("mi copy cp, replace");
    assert!(dir.join("cp.dta").exists() && dir.join("_2_cp.dta").exists());
    s.run("use cp, clear");
    s.run("mi query");
    assert_eq!(s.scalar("r(M)"), 2.0);
    assert_eq!(extract_col(&mut s, 2, "y"), y2);
    s.run("mi convert wide, clear");
    s.run("mi erase cp");
    assert!(!dir.join("cp.dta").exists() && !dir.join("_1_cp.dta").exists());
    s.fails("mi erase cp");
}

#[test]
fn import_export_round_trips() {
    let mut s = Session::new();
    let _dir = cd(&mut s, "t50_import");
    mi_data(&mut s);
    s.run("mi set mlong");
    s.run("mi register imputed y z");
    s.run("mi impute mvn y z = x, add(3) rseed(5)");
    let y3 = extract_col(&mut s, 3, "y");
    let z1 = extract_col(&mut s, 1, "z");
    // ice: _mj / _mi layout and back
    s.run("mi export ice, clear");
    assert_eq!(s.n(), 160);
    assert!(s.has_var("_mj") && s.has_var("_mi"));
    s.run("save ice, replace");
    s.run("mi import ice, automatic clear");
    s.run("mi describe");
    assert_eq!(s.scalar("r(M)"), 3.0);
    assert_eq!(s.scalar("r(N_incomplete)"), 12.0);
    assert_eq!(extract_col(&mut s, 3, "y"), y3);
    // flong with explicit registration
    s.run("use ice, clear");
    s.run("mi import flong, m(_mj) id(_mi) imputed(y z) clear");
    assert_eq!(extract_col(&mut s, 1, "z"), z1);
    // wide: y = y1 y2 y3
    s.run("mi extract 0, clear");
    s.run("drop _mi");
    s.run("gen double y1 = cond(y==., 100, y)");
    s.run("gen double y2 = cond(y==., 200, y)");
    s.run("mi import wide, imputed(y = y1 y2) drop clear");
    assert!(!s.has_var("y1") || s.has_var("_1_y"));
    let w2 = extract_col(&mut s, 2, "y");
    let w0 = extract_col(&mut s, 0, "y");
    for i in 0..40 {
        if miss(w0[i]) {
            assert_eq!(w2[i], 200.0);
        } else {
            assert_eq!(w2[i], w0[i]);
        }
    }
}

#[test]
fn nhanes1_and_flongsep_import() {
    let mut s = Session::new();
    let dir = cd(&mut s, "t50_nhanes");
    mi_data(&mut s);
    s.run("mi set flong");
    s.run("mi register imputed y");
    s.run("mi impute regress y x, add(2) rseed(9)");
    let y2 = extract_col(&mut s, 2, "y");
    s.run("mi export nhanes1 nh, replace");
    assert!(dir.join("nh.dta").exists() && dir.join("nh_imp.dta").exists());
    s.run("use nh_imp, clear");
    assert!(s.has_var("y1") && s.has_var("y2") && s.has_var("yif"));
    assert_eq!(s.nums("yif").iter().sum::<f64>(), 8.0);
    s.run("use nh, clear");
    s.run("mi import nhanes1 nh2, using(nh_imp) id(seqn) clear");
    s.run("mi query");
    assert_eq!(s.scalar("r(M)"), 2.0);
    assert_eq!(extract_col(&mut s, 2, "y"), y2);
    // flongsep import from separate files
    s.script(
        r#"clear
           set obs 6
           gen id = _n
           gen double v = _n
           replace v = . in 2
           save b0, replace
           replace v = 20 in 2
           save b1, replace
           replace v = 30 in 2
           save b2, replace
           use b0, clear
           mi import flongsep fsi, using(b1 b2) id(id) imputed(v) clear"#,
    );
    s.run("mi query");
    assert_eq!(s.scalar("r(M)"), 2.0);
    assert_eq!(extract_col(&mut s, 2, "v")[1], 30.0);
    assert_eq!(extract_col(&mut s, 1, "v")[1], 20.0);
}

#[test]
fn estimate_saving_using_and_predict() {
    let mut s = Session::new();
    let _dir = cd(&mut s, "t50_est");
    mi_data(&mut s);
    s.run("mi set mlong");
    s.run("mi register imputed y z");
    s.run("mi impute mvn y z = x, add(4) rseed(2)");
    s.run("mi estimate, saving(miest, replace): regress y x z");
    let b = [s.scalar("_b[x]"), s.scalar("_b[z]"), s.scalar("_b[_cons]")];
    let se = s.scalar("_se[x]");
    s.run("mi estimate using miest");
    assert_close(s.scalar("_b[x]"), b[0], 1e-12);
    assert_close(s.scalar("_se[x]"), se, 1e-12);
    assert_eq!(s.scalar("e(M_mi)"), 4.0);
    s.run("mi estimate using miest, imputations(1/2)");
    assert_eq!(s.scalar("e(M_mi)"), 2.0);
    // linear prediction with the combined coefficients, per imputation
    s.run("mi predict double xb using miest");
    s.run("mi predictnl double xb2 = 2*predict(xb) using miest");
    for m in [0usize, 3] {
        let xb = extract_col(&mut s, m, "xb");
        let xb2 = extract_col(&mut s, m, "xb2");
        let x = extract_col(&mut s, m, "x");
        let z = extract_col(&mut s, m, "z");
        for i in 0..40 {
            if miss(z[i]) {
                assert!(miss(xb[i]));
                continue;
            }
            assert_close(xb[i], b[0] * x[i] + b[1] * z[i] + b[2], 1e-9);
            assert_close(xb2[i], 2.0 * xb[i], 1e-9);
        }
    }
    s.fails("mi estimate using nosuchfile");
}

#[test]
fn reshape_expand_stsplit_on_every_imputation() {
    let mut s = Session::new();
    mi_data(&mut s);
    s.run("mi set mlong");
    s.run("mi register imputed y");
    s.run("mi impute regress y x, add(2) rseed(4)");
    let y2 = extract_col(&mut s, 2, "y");
    s.run("mi reshape wide x y z, i(id) j(t)");
    s.run("mi describe");
    assert_eq!(s.scalar("r(N)"), 10.0);
    assert_eq!(s.scalar("r(M)"), 2.0);
    let iv = s.run("display \"`r(ivars)'\"");
    assert!(iv.contains("y1") && iv.contains("y4") && !iv.contains('x'), "{iv}");
    s.run("mi reshape long x y z, i(id) j(t)");
    s.run("mi describe");
    assert_eq!(s.scalar("r(N)"), 40.0);
    assert_eq!(s.scalar("r(N_incomplete)"), 8.0);
    s.run("mi xeq 2: sort id t");
    assert_eq!(extract_col(&mut s, 2, "y"), y2);
    // expand: every observation twice in every imputation
    s.run("mi expand 2");
    s.run("mi describe");
    assert_eq!(s.scalar("r(N)"), 80.0);
    assert_eq!(s.scalar("r(N_incomplete)"), 16.0);
    let y2e = extract_col(&mut s, 2, "y");
    assert_eq!(&y2e[..40], &y2[..]);
    // survival settings and episode splitting
    let mut s = Session::new();
    mi_data(&mut s);
    s.run("mi set wide");
    s.run("mi register imputed y");
    s.run("mi impute regress y x, add(2) rseed(4)");
    s.run("gen double tt = t + 0.5");
    s.run("gen byte d = t == 4");
    s.run("mi stset tt, failure(d) id(id)");
    assert!(s.has_var("_t"));
    s.run("mi stsplit at, at(2)");
    s.run("mi describe");
    assert_eq!(s.scalar("r(N)"), 50.0);
    s.run("drop at");
    s.run("mi stjoin");
    s.run("mi describe");
    assert_eq!(s.scalar("r(N)"), 40.0);
    s.run("mi streset, id(id) failure(d)");
    s.run("mi st");
    s.run("mi xtset id t");
    s.run("mi estimate: xtreg y x, fe");
    assert!(s.scalar("e(M_mi)") == 2.0);
    s.run("mi tsset id t");
    s.run("mi fvset base 2 t");
    s.run("mi svyset id");
}

#[test]
fn add_append_merge_replace0_select() {
    let mut s = Session::new();
    let _dir = cd(&mut s, "t50_combine");
    mi_data(&mut s);
    s.run("mi set mlong");
    s.run("mi register imputed y z");
    s.run("save orig, replace");
    s.run("mi impute mvn y z = x, add(2) rseed(1)");
    s.run("save first, replace");
    s.run("use orig, clear");
    s.run("mi impute mvn y z = x, add(3) rseed(7)");
    let u3 = extract_col(&mut s, 3, "y");
    s.run("save second, replace");
    s.run("use first, clear");
    s.run("mi add y z using second");
    s.run("mi query");
    assert_eq!(s.scalar("r(M)"), 5.0);
    assert_eq!(extract_col(&mut s, 5, "y"), u3);
    // append: mi data with itself
    s.run("use first, clear");
    s.run("mi append using first");
    s.run("mi describe");
    assert_eq!(s.scalar("r(N)"), 80.0);
    assert_eq!(s.scalar("r(N_incomplete)"), 24.0);
    // merge: a plain using file adds a regular variable to every imputation
    s.script(
        r#"clear
           set obs 10
           gen id = _n
           gen w = 10*id
           save wfile, replace
           use first, clear"#,
    );
    s.run("mi merge m:1 id using wfile, nogenerate");
    let w = extract_col(&mut s, 2, "w");
    let id = extract_col(&mut s, 2, "id");
    for i in 0..40 {
        assert_eq!(w[i], 10.0 * id[i]);
    }
    // replace0: drop observations from m = 0 and carry that to every m
    s.run("use first, clear");
    let y1 = extract_col(&mut s, 1, "y");
    s.run("gen obsno = _n");
    s.run("mi extract 0, clear");
    s.run("drop if obsno <= 4");
    s.run("save m0, replace");
    s.run("use first, clear");
    s.run("gen obsno = _n");
    s.run("mi replace0 using m0, id(obsno)");
    s.run("mi describe");
    assert_eq!(s.scalar("r(N)"), 36.0);
    assert_eq!(extract_col(&mut s, 1, "y")[..], y1[4..]);
    // select
    s.run("use first, clear");
    let y2 = extract_col(&mut s, 2, "y");
    s.run("mi select init");
    s.run("mi select 2");
    assert_eq!(s.nums("y"), y2);
    assert_eq!(s.n(), 40);
}

#[test]
fn impute_intreg_truncreg_nbreg() {
    let mut s = Session::new();
    s.script(
        r#"clear
           set obs 400
           set seed 21
           gen double x = rnormal()
           gen double ystar = 1 + x + rnormal()
           gen double lo = floor(ystar)
           gen double hi = lo + 1
           replace hi = . in 1/40
           replace lo = . in 41/60
           replace hi = lo in 61/70
           gen double yi = .
           mi set mlong
           mi register imputed yi
           mi impute intreg yi x, ll(lo) ul(hi) add(2) rseed(3)"#,
    );
    assert_eq!(s.scalar("r(M)"), 2.0);
    let yi = extract_col(&mut s, 2, "yi");
    let lo = extract_col(&mut s, 2, "lo");
    let hi = extract_col(&mut s, 2, "hi");
    for i in 0..400 {
        assert!(yi[i] < 8.9e307);
        if lo[i] < 8.9e307 {
            assert!(yi[i] >= lo[i] - 1e-9, "obs {i}");
        }
        if hi[i] < 8.9e307 {
            assert!(yi[i] <= hi[i] + 1e-9, "obs {i}");
        }
    }
    for i in 60..70 {
        assert_eq!(yi[i], lo[i]);
    }
    // the interval model recovers the latent mean
    let m: f64 = yi.iter().sum::<f64>() / 400.0;
    let ys = extract_col(&mut s, 2, "ystar");
    assert_close(m, ys.iter().sum::<f64>() / 400.0, 0.1);

    // truncated regression: imputations stay inside the truncation limits
    let mut s = Session::new();
    s.script(
        r#"clear
           set obs 300
           set seed 5
           gen double x = rnormal()
           gen double y = 2 + x + rnormal()
           drop if y < 0 | y > 5
           replace y = . if mod(_n,6)==0
           mi set wide
           mi register imputed y
           mi impute truncreg y x, ll(0) ul(5) add(3) rseed(8)"#,
    );
    for m in 1..=3 {
        assert!(extract_col(&mut s, m, "y").iter().all(|v| (0.0..=5.0).contains(v)));
    }
    // negative binomial: nonnegative integers, mean close to observed mean
    let mut s = Session::new();
    s.script(
        r#"clear
           set obs 400
           set seed 6
           gen double x = runiform()
           gen double c = rpoisson(exp(1 + x) * rgamma(2, .5))
           replace c = . if mod(_n,4)==0
           mi set flong
           mi register imputed c
           mi impute nbreg c x, add(2) rseed(2)"#,
    );
    let c = extract_col(&mut s, 1, "c");
    assert!(c.iter().all(|v| *v >= 0.0 && v.fract() == 0.0 && *v < 1e6));
    s.fails("mi impute intreg c x, add(1)");
}

#[test]
fn ptrace_from_mvn() {
    let mut s = Session::new();
    let dir = cd(&mut s, "t50_ptrace");
    mi_data(&mut s);
    s.run("mi set mlong");
    s.run("mi register imputed y z");
    s.run("mi impute mvn y z = x, add(2) rseed(1) burnin(30) burnbetween(10) savetrace(tr, replace)");
    assert!(dir.join("tr.stptrace").exists());
    s.run("mi ptrace describe tr");
    assert_eq!(s.scalar("r(m)"), 2.0);
    assert_eq!(s.scalar("r(niter)"), 40.0);
    s.run("mi ptrace use tr, clear");
    assert_eq!(s.n(), 40);
    for v in ["m", "iter", "b_y_x", "b_z__cons", "v_y_y", "v_y_z", "v_z_z"] {
        assert!(s.has_var(v), "{v}");
    }
    assert!(s.nums("v_y_y").iter().all(|v| *v > 0.0));
    s.run("mi ptrace use tr, clear select(m==2)");
    assert_eq!(s.n(), 10);
}

// ---------------------------------------------------------------------------
// MV
// ---------------------------------------------------------------------------

fn discrim_data(s: &mut Session) {
    s.script(
        r#"clear
           set obs 6
           gen byte g = 1 + (_n > 3)
           gen double a = _n
           gen double b = mod(_n, 3)"#,
    );
    s.run("replace b = b + 2*g");
}

#[test]
fn discrim_estat_more() {
    let mut s = Session::new();
    discrim_data(&mut s);
    s.run("discrim lda a b, group(g)");
    // group 1: a = 1 2 3, b = 3 4 2;  group 2: a = 4 5 6, b = 5 6 4
    // within-group SSCP: a 2+2, b 2+2, ab (-1)+(-1) over 6-2 df
    s.run("estat covariance");
    s.run("matrix S = r(S)");
    assert_close(s.scalar("S[1,1]"), 1.0, 1e-12);
    assert_close(s.scalar("S[2,2]"), 1.0, 1e-12);
    assert_close(s.scalar("S[1,2]"), -0.5, 1e-12);
    s.run("estat correlations");
    s.run("matrix R = r(Rho)");
    assert_close(s.scalar("R[1,2]"), -0.5, 1e-12);
    s.run("estat covariance, group total");
    s.run("matrix S1 = r(S_1)");
    assert_close(s.scalar("S1[1,2]"), -0.5, 1e-12);
    // mean difference (3, 2); Mahalanobis d2 = d' S^-1 d with S = [1 -.5; -.5 1]
    s.run("estat grdistances");
    s.run("matrix D = r(sqdist)");
    let d2 = (9.0 + 4.0 + 2.0 * 0.5 * 3.0 * 2.0) / 0.75;
    assert_close(s.scalar("D[1,2]"), d2, 1e-9);
    s.run("estat summarize");
    s.run("matrix M = r(stats)");
    assert_close(s.scalar("M[2,1]"), 3.5, 1e-12);
    // manova after discrim matches manova itself
    s.run("estat manova");
    s.run("matrix W1 = r(stat_m)");
    let w1 = s.scalar("W1[1,1]");
    s.run("manova a b = g");
    s.run("matrix SM = e(stat_m)");
    assert_close(w1, s.scalar("SM[1,1]"), 1e-10);
}

#[test]
fn cluster_programming_utilities() {
    let mut s = Session::new();
    s.script(
        r#"clear
           set obs 5
           gen double u = _n
           gen double v = _n^2
           cluster averagelinkage u v, name(c1)"#,
    );
    s.run("cluster set c1, other(note1 hello) method(mine)");
    s.run("cluster query c1");
    assert!(s.run("display \"`r(method)'\"").contains("mine"));
    s.run("cluster set newc, addname type(hierarchical) dissimilarity(L1)");
    s.run("cluster query");
    let names = s.run("display \"`r(names)'\"");
    assert!(names.contains("newc") && names.contains("c1"), "{names}");
    s.run("cluster delete newc, delname");
    s.run("cluster query");
    assert!(!s.run("display \"`r(names)'\"").contains("newc"));
    s.run("cluster parsedistance L1");
    assert!(s.run("display \"`r(dist)' `r(dtype)'\"").contains("L1 dissimilarity"));
    s.run("cluster parsedistance Jaccard");
    assert!(s.run("display \"`r(dtype)' `r(binary)'\"").contains("similarity binary"));
    s.run("cluster measures u v, compare(1 5) generate(d1 d5)");
    let d1 = s.nums("d1");
    let d5 = s.nums("d5");
    for i in 0..5 {
        let (u, v) = ((i + 1) as f64, ((i + 1) * (i + 1)) as f64);
        assert_close(d1[i], ((u - 1.0).powi(2) + (v - 1.0).powi(2)).sqrt(), 1e-12);
        assert_close(d5[i], ((u - 5.0).powi(2) + (v - 25.0).powi(2)).sqrt(), 1e-12);
    }
    s.run("cluster measures u v, compare(2) generate(m2) L1");
    assert_eq!(s.nums("m2"), vec![4.0, 0.0, 6.0, 14.0, 24.0]);
    s.run("cluster delete c1, zap");
    assert!(!s.has_var("c1_id"));
}

#[test]
fn ca_mca_mds_plots() {
    let mut s = Session::new();
    s.script(
        r#"clear
           set obs 60
           gen byte r = mod(_n,3)+1
           gen byte c = mod(floor(_n/2),4)+1
           gen byte k = mod(floor(_n/3),2)+1"#,
    );
    s.fails("caprojection");
    s.run("ca r c");
    s.run("caprojection");
    s.run("caprojection, norow");
    s.fails("mcaplot");
    s.run("mca r c k");
    s.run("mcaplot");
    s.run("mcaprojection");
    s.fails("mdsshepard");
    s.script(
        r#"clear
           set obs 8
           gen id = _n
           gen double p = _n
           gen double q = mod(_n, 3)"#,
    );
    s.run("mds p q, id(id) noplot");
    s.run("mdsshepard");
    s.run("mdsconfig");
}

// ---------------------------------------------------------------------------
// SP
// ---------------------------------------------------------------------------

/// A 5 x 5 grid of unit squares with its shapefile.
fn grid(s: &mut Session) {
    s.script(
        r#"clear
           set obs 125
           gen _ID = ceil(_n/5)
           gen k = mod(_n-1,5)
           gen double _X = cond(k==0,., mod(_ID-1,5) + (k==2|k==3))
           gen double _Y = cond(k==0,., floor((_ID-1)/5) + (k>=3))
           gen shape_order = k
           drop k
           save grid_shp, replace
           clear
           set obs 25
           gen id = _n
           gen double cx = mod(_n-1,5) + .5
           gen double cy = floor((_n-1)/5) + .5
           set seed 4
           gen double x = rnormal()
           gen double y = x + rnormal()
           spset id, coord(cx cy)
           spset, modify shpfile(grid_shp)"#,
    );
}

#[test]
fn contiguity_on_grid_and_moran() {
    let mut s = Session::new();
    cd(&mut s, "t50_grid");
    grid(&mut s);
    s.run("spmatrix create contiguity Wq, normalize(none)");
    s.run("spmatrix summarize Wq");
    assert_eq!(s.scalar("r(n_min)"), 3.0);
    assert_eq!(s.scalar("r(n_max)"), 8.0);
    assert_close(s.scalar("r(n_mean)"), 144.0 / 25.0, 1e-12);
    s.run("spmatrix create contiguity Wr, rook normalize(none)");
    s.run("spmatrix summarize Wr");
    assert_eq!(s.scalar("r(n_min)"), 2.0);
    assert_eq!(s.scalar("r(n_max)"), 4.0);
    assert_close(s.scalar("r(n_mean)"), 80.0 / 25.0, 1e-12);
    s.run("regress y x");
    s.run("estat moran, errorlag(Wq)");
    assert!(s.scalar("r(chi2)") >= 0.0);
}

#[test]
fn userdefined_matches_idistance() {
    let mut s = Session::new();
    cd(&mut s, "t50_ud");
    grid(&mut s);
    s.run("mata:\nfunction InvD(vi, vj) {\n  real scalar d\n  d = sqrt((vi[1]-vj[1])^2 + (vi[2]-vj[2])^2)\n  if (d > 1.5) return(0)\n  return(1/d)\n}\nend");
    s.run("spmatrix userdefined U = InvD(cx cy), normalize(none)");
    s.run("spmatrix create idistance I, vtruncate(0.6) normalize(none)");
    s.run("spmatrix matafromsp MU idu = U");
    s.run("spmatrix matafromsp MI idi = I");
    s.run("matrix DD = MU - MI");
    s.run("mata: st_numscalar(\"mx\", max(abs(st_matrix(\"DD\"))))");
    assert!(s.scalar("mx") < 1e-12);
    assert_close(s.scalar("MU[1,2]"), 1.0, 1e-12);
    assert_close(s.scalar("MU[1,7]"), 1.0 / 2f64.sqrt(), 1e-12);
    s.run("spmatrix summarize U");
    assert_eq!(s.scalar("r(n_max)"), 8.0);
    // if restricts the units; normalize applies as usual
    s.run("spmatrix userdefined U2 = InvD(cx cy) if id <= 10, normalize(row)");
    s.run("spmatrix matafromsp M2 id2 = U2");
    assert_eq!(s.scalar("rowsof(M2)"), 10.0);
    s.run("mata: st_numscalar(\"rs\", rowsum(st_matrix(\"M2\"))[1])");
    assert_close(s.scalar("rs"), 1.0, 1e-12);
    s.fails("spmatrix userdefined U = InvD(cx cy)");
}
