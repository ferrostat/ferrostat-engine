//! r(table) after estimation, HC2/HC3 variances, `mean, over()` coefficient
//! names and the `contrast` post-estimation command.

use ferrostat_tests::*;

const TOL: f64 = 1e-8;

#[test]
fn r_table_after_regress_and_replay() {
    let mut s = Session::with_fixture();
    s.run("regress y x z");
    s.run("matrix P = r(table)");
    assert_eq!(s.scalar("rowsof(P)"), 9.0);
    assert_eq!(s.scalar("colsof(P)"), 3.0);
    assert_close(s.scalar("el(P,1,1)"), s.scalar("_b[x]"), TOL);
    assert_close(s.scalar("el(P,2,1)"), s.scalar("_se[x]"), TOL);
    assert_close(s.scalar("el(P,3,1)"), s.scalar("_b[x]/_se[x]"), 1e-6);
    assert_close(s.scalar("el(P,4,2)"), s.scalar("2*ttail(e(df_r), abs(_b[z]/_se[z]))"), 1e-6);
    assert_close(s.scalar("el(P,5,1)"), s.scalar("_b[x] - invttail(e(df_r),0.025)*_se[x]"), 1e-6);
    assert_close(s.scalar("el(P,6,1)"), s.scalar("_b[x] + invttail(e(df_r),0.025)*_se[x]"), 1e-6);
    assert_eq!(s.scalar("el(P,7,1)"), 9.0);
    assert_eq!(s.scalar("el(P,9,1)"), 0.0);
    assert_eq!(s.scalar("r(level)"), 95.0);
    // subscripting the returned matrix directly
    s.run("matrix A = J(1,1,.)");
    s.run("matrix A[1,1] = r(table)[1,2]");
    assert_close(s.scalar("el(A,1,1)"), s.scalar("_b[z]"), TOL);
    // replay restores r(table)
    s.run("summarize x");
    s.run("regress");
    assert_close(s.scalar("el(r(table),1,1)"), s.scalar("_b[x]"), TOL);
}

#[test]
fn r_table_base_levels_and_normal_models() {
    let mut s = Session::with_fixture();
    s.run("regress y i.group");
    // columns include the base level
    assert_eq!(s.scalar("colsof(r(table))"), 4.0);
    assert_eq!(s.scalar("el(r(table),1,1)"), 0.0);
    assert_eq!(s.scalar("missing(el(r(table),2,1))"), 1.0);
    assert_close(s.scalar("el(r(table),1,2)"), s.scalar("_b[2.group]"), TOL);
    s.run("logit d x");
    assert_eq!(s.scalar("missing(el(r(table),7,1))"), 1.0);
    assert_close(s.scalar("el(r(table),8,1)"), 1.959963984540054, 1e-7);
    s.run("logit d x, or");
    assert_close(s.scalar("el(r(table),1,1)"), s.scalar("exp(_b[x])"), 1e-8);
    assert_eq!(s.scalar("el(r(table),9,1)"), 1.0);
}

/// HC2/HC3 computed directly for a regression on one regressor.
fn hc_manual(power: i32) -> [f64; 2] {
    let n = X.len();
    let (sx, sxx) = (X.iter().sum::<f64>(), X.iter().map(|x| x * x).sum::<f64>());
    let det = n as f64 * sxx - sx * sx;
    let a = [[sxx / det, -sx / det], [-sx / det, n as f64 / det]];
    let sy: f64 = Y.iter().sum();
    let sxy: f64 = X.iter().zip(&Y).map(|(x, y)| x * y).sum();
    let b0 = a[0][0] * sy + a[0][1] * sxy;
    let b1 = a[1][0] * sy + a[1][1] * sxy;
    let mut meat = [[0.0; 2]; 2];
    for t in 0..n {
        let xv = [1.0, X[t]];
        let h: f64 = (0..2).map(|p| (0..2).map(|q| xv[p] * a[p][q] * xv[q]).sum::<f64>()).sum();
        let e = Y[t] - b0 - b1 * X[t];
        let f = e * e / (1.0 - h).powi(power);
        for p in 0..2 {
            for q in 0..2 {
                meat[p][q] += f * xv[p] * xv[q];
            }
        }
    }
    let mut v = [[0.0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            v[i][j] = (0..2).map(|p| (0..2).map(|q| a[i][p] * meat[p][q] * a[q][j]).sum::<f64>()).sum();
        }
    }
    [v[1][1].sqrt(), v[0][0].sqrt()]
}

#[test]
fn regress_vce_hc2_hc3() {
    let mut s = Session::with_fixture();
    for (opt, pw) in [("hc2", 1), ("hc3", 2)] {
        s.run(&format!("regress y x, vce({opt})"));
        let want = hc_manual(pw);
        assert_close(s.scalar("_se[x]"), want[0], 1e-9);
        assert_close(s.scalar("_se[_cons]"), want[1], 1e-9);
        assert_eq!(s.run("display e(vce)").trim(), opt);
        assert_close(s.scalar("el(r(table),2,1)"), want[0], 1e-9);
    }
    s.run("regress y x, vce(cluster group)");
    assert_eq!(s.scalar("e(N_clust)"), 3.0);
    s.fails("regress y x, vce(hc4)");
}

#[test]
fn mean_over_names_and_tests() {
    let mut s = Session::with_fixture();
    s.run("mean y x, over(d)");
    s.run("matrix b = e(b)");
    assert_eq!(s.run("display \"`: colnames b'\"").trim(), "c.y@0.d c.y@1.d c.x@0.d c.x@1.d");
    s.run("summarize y if d==1");
    let m1 = s.scalar("r(mean)");
    let se1 = s.scalar("r(sd)/sqrt(r(N))");
    s.run("mean y x, over(d)");
    assert_close(s.scalar("_b[y@1.d]"), m1, TOL);
    assert_close(s.scalar("_b[c.y@1.d]"), m1, TOL);
    assert_close(s.scalar("_se[y@1.d]"), se1, TOL);
    assert_close(s.scalar("el(r(table),1,2)"), m1, TOL);
    assert_eq!(s.scalar("e(df_r)"), 11.0);
    s.run("lincom y@1.d - y@0.d");
    let diff = s.scalar("r(estimate)");
    assert_close(diff, s.scalar("_b[y@1.d] - _b[y@0.d]"), TOL);
    s.run("test y@0.d = y@1.d");
    let p = s.scalar("r(p)");
    assert!(p > 0.0 && p < 1.0);
    // two grouping variables
    s.run("mean y, over(d group)");
    s.run("matrix b = e(b)");
    assert!(s.run("display \"`: colnames b'\"").contains("c.y@0.d#1.group"));
}

#[test]
fn mean_over_legacy_names() {
    let mut s = Session::with_fixture();
    s.run("version 15");
    s.run("mean y, over(d)");
    s.run("matrix b = e(b)");
    assert_eq!(s.run("display \"`: colfullnames b'\"").trim(), "y:no y:yes");
    s.run("test no = yes");
    assert!(s.scalar("r(p)") < 1.0);
    s.run("version 18");
    s.run("mean y, over(d)");
    s.run("matrix b = e(b)");
    assert_eq!(s.run("display \"`: colnames b'\"").trim(), "c.y@0.d c.y@1.d");
}

#[test]
fn contrast_reference_within_groups() {
    let mut s = Session::with_fixture();
    s.run("regress y i.d##i.group");
    let b1 = s.scalar("_b[1.d]");
    let b2 = s.scalar("_b[1.d] + _b[1.d#2.group]");
    s.run("lincom 1.d + 1.d#2.group");
    let se2 = s.scalar("r(se)");
    let out = s.run("contrast r.d@i.group");
    assert!(out.contains("Joint"), "{out}");
    assert!(out.contains("Denominator"), "{out}");
    assert_close(s.scalar("el(r(b),1,1)"), b1, 1e-9);
    s.run("contrast r.d@i.group, post effects");
    s.run("matrix b = e(b)");
    assert_eq!(s.run("display \"`: colnames b'\"").trim(), "r1vs0.d@1.group r1vs0.d@2.group r1vs0.d@3.group");
    assert_close(s.scalar("_b[r1vs0.d@2.group]"), b2, 1e-9);
    assert_close(s.scalar("_se[r1vs0.d@2.group]"), se2, 1e-9);
    assert_eq!(s.run("display e(cmd)").trim(), "contrast");
    s.run("estimates store c1");
}

#[test]
fn contrast_operators_and_tests() {
    let mut s = Session::with_fixture();
    s.run("regress y x i.group");
    // a plain term: joint test equals testparm
    s.run("contrast group");
    let f = s.scalar("el(r(F),1,1)");
    s.run("testparm i.group");
    assert_close(f, s.scalar("r(F)"), 1e-8);
    s.run("contrast r.group");
    assert_close(s.scalar("el(r(b),1,2)"), s.scalar("_b[3.group]"), 1e-9);
    s.run("contrast a.group");
    assert_close(s.scalar("el(r(b),1,1)"), s.scalar("-_b[2.group]"), 1e-9);
    assert_close(s.scalar("el(r(b),1,2)"), s.scalar("_b[2.group]-_b[3.group]"), 1e-9);
    s.run("contrast g.group");
    assert_close(s.scalar("el(r(b),1,1)"), s.scalar("-(_b[2.group]+_b[3.group])/3"), 1e-9);
    s.fails("contrast r.x");
    s.run("logit d x i.group");
    let out = s.run("contrast group");
    assert!(out.contains("chi2"), "{out}");
}
