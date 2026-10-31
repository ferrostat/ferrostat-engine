//! Tree-ensemble learners behind the `h2oml` commands (native backend):
//! gradient boosting, random forests, TreeSHAP, metrics, grids, frames
//! and the postestimation commands, checked on synthetic data.

use ferrostat_cmd::rng::Rng;
use ferrostat_cmd::w7f::tree::{grow, shap_bruteforce, GrowParams};
use ferrostat_tests::*;

fn data(s: &mut Session, n: usize) {
    s.script(&format!(
        "clear
         set obs {n}
         set seed 7
         gen double x1 = runiform()
         gen double x2 = rnormal()
         gen double x3 = runiform()
         gen double y = 2*(x1>.5) + x2 + .3*rnormal()
         gen byte b = (y + .5*rnormal()) > 1
         gen byte c = 1 + (x1>.33) + (x1>.66) + (x3>.5)"
    ));
}

#[test]
fn stump_gbm_reproduces_group_means() {
    let mut s = Session::new();
    s.script(
        "set obs 40
         gen g = _n > 15
         gen double y = cond(g, 5, 1) + mod(_n, 4)",
    );
    s.run("h2oml gbregress y g, ntrees(1) maxdepth(1) lrate(1) minobsleaf(1) h2orseed(1)");
    s.run("h2omlpredict p");
    s.run("bysort g: egen double m = mean(y)");
    let p = s.nums("p");
    let m = s.nums("m");
    assert_all_close(&p, &m, 1e-10);
    // three groups need depth 2
    s.script(
        "gen k = mod(_n, 3)
         replace y = 10*k + mod(_n, 5)
         h2oml gbregress y k, ntrees(1) maxdepth(2) lrate(1) minobsleaf(1) h2orseed(1)
         h2omlpredict p2
         bysort k: egen double m2 = mean(y)",
    );
    assert_all_close(&s.nums("p2"), &s.nums("m2"), 1e-10);
}

#[test]
fn random_forest_fits_noiseless_step() {
    let mut s = Session::new();
    s.script(
        "set obs 100
         gen double x = cond(_n <= 50, .4*_n/50, .6 + .4*(_n-50)/50)
         gen double y = 3*(x > .5)
         gen double z = mod(_n, 7)
         h2oml rfregress y x, ntrees(10) h2orseed(4)
         h2omlpredict p",
    );
    assert_all_close(&s.nums("p"), &s.nums("y"), 1e-12);
    // classification forest on the same step
    s.script(
        "gen byte cl = x > .5
         h2oml rfbinclass cl x, ntrees(10) h2orseed(4)
         h2omlpredict pc, pr
         h2omlpredict cc",
    );
    let pc = s.nums("pc");
    let cl = s.nums("cl");
    assert_all_close(&pc, &cl, 1e-12);
    assert_eq!(s.nums("cc"), cl);
    assert_close(s.scalar("e(auc)"), 1.0, 1e-12);
}

#[test]
fn treeshap_matches_exact_shapley_values() {
    let mut rng = Rng::new(3);
    let n = 300;
    let p = 4;
    let x: Vec<Vec<f64>> = (0..p).map(|_| (0..n).map(|_| rng.uniform()).collect()).collect();
    let y: Vec<f64> = (0..n).map(|i| 2.0 * x[0][i] + (x[1][i] > 0.5) as i32 as f64 * x[2][i] + 0.1 * rng.normal()).collect();
    let rows: Vec<usize> = (0..n).collect();
    let prm = GrowParams { max_depth: 5, min_leaf: 3.0, mtry: None, scale: 1.0 };
    let t = grow(&x, &rows, &[y], &vec![1.0; n], &prm, &mut rng);
    assert!(t.depth() >= 3);
    for i in [0usize, 17, 123, 299] {
        let xr = |j: usize| x[j][i];
        let mut phi = vec![0.0; p];
        t.shap(&xr, 0, &mut phi);
        let exact = shap_bruteforce(&t, &xr, 0, p);
        assert_all_close(&phi, &exact, 1e-10);
        let sum: f64 = phi.iter().sum();
        assert_close(sum, t.leaf(&xr).value[0] - t.expected(0), 1e-10);
    }
}

#[test]
fn shap_contributions_add_up_to_predictions() {
    let mut s = Session::new();
    data(&mut s, 150);
    s.script(
        "h2oml gbregress y x1 x2 x3, ntrees(20) h2orseed(1)
         h2omlpredict yhat
         h2omlpredict s, contrib
         gen double d = s_x1 + s_x2 + s_x3 + s_bias - yhat",
    );
    for v in s.nums("d") {
        assert!(v.abs() < 1e-9, "{v}");
    }
    // the bias is the same for all observations
    let b = s.nums("s_bias");
    assert!(b.iter().all(|v| (v - b[0]).abs() < 1e-12));
    // binary boosting: contributions on the log-odds scale
    s.script(
        "h2oml gbbinclass b x1 x2 x3, ntrees(10) h2orseed(1)
         h2omlpredict pb, pr
         h2omlpredict t, contrib
         gen double d2 = t_x1 + t_x2 + t_x3 + t_bias - logit(pb)",
    );
    for v in s.nums("d2") {
        assert!(v.abs() < 1e-9, "{v}");
    }
    // forests: contributions on the probability/value scale
    s.script(
        "h2oml rfregress y x1 x2 x3, ntrees(8) h2orseed(2)
         h2omlpredict rf
         h2omlpredict u, contrib
         gen double d3 = u_x1 + u_x2 + u_x3 + u_bias - rf",
    );
    for v in s.nums("d3") {
        assert!(v.abs() < 1e-9, "{v}");
    }
    // multiclass boosting: per-class contributions give the class margins
    s.script(
        "gen byte c3 = 1 + (x1>.4) + (x2>.3)
         h2oml gbmulticlass c3 x1 x2 x3, ntrees(6) h2orseed(1)
         h2omlpredict q1 q2 q3, pr
         h2omlpredict a, contrib class(1)
         h2omlpredict b, contrib class(2)
         h2omlpredict c, contrib class(3)
         gen double f1 = exp(a_x1 + a_x2 + a_x3 + a_bias)
         gen double f2 = exp(b_x1 + b_x2 + b_x3 + b_bias)
         gen double f3 = exp(c_x1 + c_x2 + c_x3 + c_bias)
         gen double d4 = f2/(f1+f2+f3) - q2",
    );
    for v in s.nums("d4") {
        assert!(v.abs() < 1e-9, "{v}");
    }
    s.fails("h2omlpredict zz, contrib");
    s.run("h2omlgraph shapsummary, class(3)");
    s.run("h2oml rfregress y x1 x2 x3, ntrees(8) h2orseed(2)");
    s.run("h2omlgraph shapsummary");
    s.run("h2omlgraph shapvalues, obs(3)");
    assert_close(s.scalar("el(r(shapvalues),1,1) + el(r(shapvalues),1,2) + el(r(shapvalues),1,3) + r(expected)"), s.nums("rf")[2], 1e-9);
}

#[test]
fn perfect_classifier_and_confusion_counts() {
    let mut s = Session::new();
    data(&mut s, 120);
    s.script(
        "gen byte perfect = x1 > .5
         h2oml gbbinclass perfect x1 x2, ntrees(5) minobsleaf(1) h2orseed(1)",
    );
    assert_close(s.scalar("e(auc)"), 1.0, 1e-12);
    assert_close(s.scalar("e(mpce)"), 0.0, 1e-12);
    s.run("h2omlestat confmatrix");
    assert_close(s.scalar("el(r(confmatrix),1,2) + el(r(confmatrix),2,1)"), 0.0, 0.0);
    // counts add up to the sample, for binary and multiclass models
    s.run("h2oml rfbinclass b x1 x2 x3, ntrees(10) h2orseed(3)");
    s.run("h2omlestat confmatrix");
    let tot = s.scalar("el(r(confmatrix),1,1)+el(r(confmatrix),1,2)+el(r(confmatrix),2,1)+el(r(confmatrix),2,2)");
    assert_eq!(tot, 120.0);
    s.run("h2omlestat confmatrix, threshold(.5)");
    assert_eq!(s.scalar("r(threshold)"), 0.5);
    let out = s.run("h2omlestat threshmetric");
    assert!(out.contains("f1") && out.contains("absolute mcc"), "{out}");
    s.run("h2oml gbmulticlass c x1 x2 x3, ntrees(5) h2orseed(3)");
    s.run("h2omlestat confmatrix");
    let mut sum = 0.0;
    let k = s.scalar("rowsof(r(confmatrix))") as usize;
    assert_eq!(k, 4);
    for a in 1..=k {
        for b in 1..=k {
            sum += s.scalar(&format!("el(r(confmatrix),{a},{b})"));
        }
    }
    assert_eq!(sum, 120.0);
    s.run("h2omlestat hitratio");
    assert_close(s.scalar("el(r(hitratio),4,1)"), 1.0, 1e-12);
    assert!(s.scalar("el(r(hitratio),1,1)") <= s.scalar("el(r(hitratio),2,1)"));
    s.run("h2omlestat aucmulticlass");
    assert!(s.scalar("r(auc_macro)") > 0.5);
    // probabilities sum to one
    s.run("h2omlpredict q1 q2 q3 q4, pr");
    s.run("gen double qs = q1 + q2 + q3 + q4");
    for v in s.nums("qs") {
        assert_close(v, 1.0, 1e-12);
    }
}

#[test]
fn same_seed_same_predictions() {
    let mut s = Session::new();
    data(&mut s, 100);
    s.script(
        "h2oml rfregress y x1 x2 x3, ntrees(6) samprate(.8) h2orseed(11)
         h2omlpredict a
         h2oml rfregress y x1 x2 x3, ntrees(6) samprate(.8) h2orseed(11)
         h2omlpredict b2
         h2oml rfregress y x1 x2 x3, ntrees(6) samprate(.8) h2orseed(12)
         h2omlpredict c2
         h2oml gbregress y x1 x2 x3, ntrees(6) samprate(.7) predsamprate(.5) h2orseed(5)
         h2omlpredict g1
         h2oml gbregress y x1 x2 x3, ntrees(6) samprate(.7) predsamprate(.5) h2orseed(5)
         h2omlpredict g2",
    );
    assert_eq!(s.nums("a"), s.nums("b2"));
    assert_ne!(s.nums("a"), s.nums("c2"));
    assert_eq!(s.nums("g1"), s.nums("g2"));
    // without h2orseed(), set seed controls the draws
    s.script(
        "set seed 99
         h2oml rfregress y x1 x2 x3, ntrees(4)
         h2omlpredict d1
         set seed 99
         h2oml rfregress y x1 x2 x3, ntrees(4)
         h2omlpredict d2",
    );
    assert_eq!(s.nums("d1"), s.nums("d2"));
}

#[test]
fn cross_validation_grid_and_selection() {
    let mut s = Session::new();
    data(&mut s, 120);
    s.run("h2oml gbregress y x1 x2 x3, ntrees(10) h2orseed(1) cv(4)");
    assert_eq!(s.scalar("e(cv_folds)"), 4.0);
    assert!(s.scalar("e(cv_mse)") > s.scalar("e(mse)"));
    s.run("h2omlestat cvsummary");
    assert_eq!(s.scalar("colsof(r(cvsummary))"), 6.0);
    let out = s.run("h2oml gbregress y x1 x2 x3, ntrees(4 8) maxdepth(1/3) h2orseed(1) cv(3)");
    assert!(out.contains("Grid summary"), "{out}");
    assert_eq!(s.scalar("rowsof(r(gridsummary))"), 6.0);
    // sorted by cross-validated deviance, best first and selected
    let m1 = s.scalar("el(r(gridsummary),1,4)");
    let m6 = s.scalar("el(r(gridsummary),6,4)");
    assert!(m1 <= m6);
    assert_close(s.scalar("e(cv_deviance)"), m1, 1e-12);
    s.run("h2omlestat gridsummary");
    let nt = s.scalar("el(r(gridsummary),3,2)");
    let md = s.scalar("el(r(gridsummary),3,3)");
    s.run("h2omlselect, id(3)");
    assert_eq!(s.scalar("e(ntrees)"), nt);
    assert_eq!(s.scalar("e(maxdepth)"), md);
    s.run("h2omlexplore");
    s.run("h2omlexplore, ntrees(3)");
    // random grid limited to 3 models
    s.run("h2oml rfregress y x1 x2 x3, ntrees(3 5) maxdepth(2 4 6) tune(grid(random) maxmodels(3)) h2orseed(2)");
    assert_eq!(s.scalar("rowsof(r(gridsummary))"), 3.0);
}

#[test]
fn frames_validation_and_postestimation() {
    let mut s = Session::new();
    data(&mut s, 200);
    let out = s.run("h2o init");
    assert!(out.contains("native"), "{out}");
    s.script(
        "_h2oframe put, into(all) current
         _h2oframe split all, into(train valid) split(.7 .3) rseed(1)
         _h2oframe change train",
    );
    s.run("h2oml gbbinclass b x1 x2 x3, ntrees(30) validframe(valid) stop(3) stoptolerance(0.01) h2orseed(1)");
    let nv = s.scalar("e(N_valid)");
    let nt = s.scalar("e(N)");
    assert_eq!(nv + nt, 200.0);
    assert!(s.scalar("e(ntrees_actual)") <= 30.0);
    assert!(s.scalar("e(valid_auc)") > 0.5);
    s.run("h2omlestat confmatrix, valid");
    assert_eq!(s.scalar("r(N)"), nv);
    s.run("h2omlpostestframe valid");
    s.run("h2omlestat metrics");
    assert_close(s.scalar("r(test_auc)"), s.scalar("e(valid_auc)"), 1e-12);
    s.run("h2omlpostestframe _reset");
    // predictions go to the current frame, get brings it back
    s.run("h2omlpredict pv, pr");
    assert!(!s.has_var("pv"));
    s.run("_h2oframe get train, clear");
    assert!(s.has_var("pv"));
    assert_eq!(s.n() as f64, nt);
    s.run("_h2oframe unset");
    // stored results, gof and graphs
    s.run("h2omlest store m1");
    s.run("h2oml rfbinclass b x1 x2 x3, ntrees(10) h2orseed(1)");
    s.run("h2omlest store m2");
    let out = s.run("h2omlest dir");
    assert!(out.contains("m1") && out.contains("m2"), "{out}");
    s.run("h2omlgof m1 m2");
    assert_eq!(s.scalar("colsof(r(gof))"), 2.0);
    s.run("h2omlest restore m1");
    assert_eq!(s.run("display \"`e(subcmd)'\"").trim(), "gbbinclass");
    for g in ["varimp", "permimp", "pdp x1", "ice x1", "roc", "prcurve", "scorehistory", "shapsummary", "shapvalues, obs(2)"] {
        s.run(&format!("h2omlgraph {g}"));
    }
    let out = s.run("h2omltree, tree(1)");
    assert!(out.contains("split on"), "{out}");
    s.run("h2oml gbregress y x1 x2 x3, ntrees(5) h2orseed(1)");
    s.run("h2omlgraph rvfplot");
    s.run("h2omlgraph rvpplot x1");
    s.fails("h2omlgraph roc");
    s.fails("h2oml gbregress y x1, bogus(1)");
}
