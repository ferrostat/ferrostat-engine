//! Remaining base-reference items: tables of command results, display and
//! random-number settings, ROC regression, demand systems and programming
//! helpers. Checks use synthetic data with closed forms or equalities with
//! other commands.

use ferrostat_tests::*;

fn data(s: &mut Session, n: usize) {
    s.script(&format!(
        "set obs {n}
         gen double x = mod(_n*13, 17)/4 - 2
         gen double z = mod(_n*7, 11)/3
         gen double u = (mod(_n*53, 97) + 0.5)/97
         gen double y = 1 + 0.5*x - 0.3*z + invnormal(u)
         gen g = mod(_n, 3) + 1"
    ));
}

// ---- table, command() -------------------------------------------------------

#[test]
fn table_command_collects_coefficients() {
    let mut s = Session::new();
    data(&mut s, 80);
    let out = s.run("table () (command result), command(regress y x) command(_r_b _r_se _r_p: regress y x z)");
    assert!(out.contains("regress y x z") && out.contains("Intercept") && out.contains("95% CI"), "{out}");
    s.run("quietly regress y x z");
    let (b, se) = (s.scalar("_b[z]"), s.scalar("_se[z]"));
    // the cells are the estimates of each command
    let line = out.lines().find(|l| l.trim_start().starts_with("z ")).unwrap().to_string();
    let v = numbers(&line);
    assert_close(v[0], b, 1e-5 * b.abs().max(1.0));
    assert_close(v[1], se, 1e-5);
    // the collection is current: style and export work on it
    s.run("collect style cell result[_r_b], nformat(%6.3f)");
    let p = temp_dir("t57_table").join("t.md");
    s.run(&format!("collect export \"{}\", replace", p.display()));
    let md = std::fs::read_to_string(&p).unwrap();
    assert!(md.contains(&format!("{:.3}", b)), "{md}");
}

#[test]
fn table_command_hypothesis_tests() {
    let mut s = Session::new();
    data(&mut s, 80);
    s.run("quietly regress y x z");
    let out = s.run("table (command) (result), command(r(F) r(p): test x z) command(F p: test z)");
    s.run("quietly test z");
    let (f, p) = (s.scalar("r(F)"), s.scalar("r(p)"));
    let line = out.lines().find(|l| l.trim_start().starts_with("test z")).unwrap().to_string();
    let v = numbers(&line);
    assert_close(v[0], f, 1e-5 * f.max(1.0));
    assert_close(v[1], p, 1e-5);
    // z-statistic equals b/se and the CI bounds are b -/+ t*se
    let out = s.run("table () (result), command(_r_z _r_lb _r_ub: regress y x z) nformat(%12.6f)");
    s.run("quietly regress y x z");
    let (b, se) = (s.scalar("_b[x]"), s.scalar("_se[x]"));
    let t = s.scalar("invttail(e(df_r), .025)");
    let line = out.lines().find(|l| l.trim_start().starts_with("x ")).unwrap().to_string();
    let v = numbers(&line);
    assert_close(v[0], b / se, 1e-5);
    assert_close(v[1], b - t * se, 1e-5);
    assert_close(v[2], b + t * se, 1e-5);
}

// ---- display settings ----------------------------------------------------------

fn interaction_data(s: &mut Session) {
    data(s, 60);
    s.script(
        "gen a = mod(_n,3)
         gen b = mod(_n,2)
         replace b = 0 if a==2",
    );
}

#[test]
fn cformat_pformat_sformat_change_coefficient_tables() {
    let mut s = Session::new();
    data(&mut s, 60);
    s.run("set cformat %7.3f");
    s.run("set pformat %5.2f");
    s.run("set sformat %6.3f");
    let out = s.run("regress y x z");
    let (b, se) = (s.scalar("_b[x]"), s.scalar("_se[x]"));
    let line = out.lines().find(|l| l.trim_start().starts_with("x |")).unwrap().to_string();
    assert!(line.contains(&format!("{:.3}", b)) && line.contains(&format!("{:.3}", se)), "{line}");
    assert!(line.contains(&format!("{:.3}", b / se)), "{line}");
    assert_eq!(s.run("display c(cformat)").trim(), "%7.3f");
    // margins tables follow the same settings
    let out = s.run("margins");
    s.run("quietly summarize y");
    let ybar = s.scalar("r(mean)");
    assert!(out.contains(&format!(" {:.3} ", ybar)), "{out}");
    // an empty value restores the default
    s.run("set cformat");
    let out = s.run("regress y x z");
    assert!(!out.lines().find(|l| l.trim_start().starts_with("x |")).unwrap().contains(&format!("{:.3} ", b)));
    assert_eq!(s.run("display \"[\" c(cformat) \"]\"").trim(), "[]");
    // widths are limited, formats validated
    s.fails("set cformat %12.3f");
    s.fails("set pformat %9.3f");
    s.fails("set sformat %s");
}

#[test]
fn showbaselevels_and_emptycells() {
    let mut s = Session::new();
    interaction_data(&mut s);
    let out = s.run("regress y i.a##i.b x");
    assert!(!out.contains("(base)") && out.contains("(empty)"), "{out}");
    let k = s.scalar("colsof(e(b))");
    s.run("set showbaselevels on");
    let out = s.run("regress y i.a##i.b x");
    assert!(out.contains("(base)") && out.contains("(empty)"), "{out}");
    assert_eq!(s.run("display c(showbaselevels)").trim(), "on");
    s.fails("set showbaselevels maybe");
    // dropping empty cells removes them from e(b) without changing the fit
    let rss = s.scalar("e(rss)");
    s.run("set emptycells drop");
    let out = s.run("regress y i.a##i.b x");
    assert!(!out.contains("(empty)"), "{out}");
    assert_close(s.scalar("colsof(e(b))"), k - 1.0, 0.0);
    assert_close(s.scalar("e(rss)"), rss, 1e-8);
    assert_eq!(s.run("display c(emptycells)").trim(), "drop");
    s.fails("set emptycells sometimes");
}

// ---- random-number generators ------------------------------------------------------

#[test]
fn rng_kinds_streams_and_state() {
    let mut s = Session::new();
    s.run("set obs 500");
    s.run("set seed 42");
    s.run("gen double u0 = runiform()");
    // selecting the default generator keeps its sequence
    s.run("set rng default");
    s.run("set seed 42");
    s.run("gen double u1 = runiform()");
    assert_eq!(s.nums("u0"), s.nums("u1"));
    assert_eq!(s.run("display c(rng_current)").trim(), "mt64");
    // the state string restores the generator exactly, mid-sequence
    s.run("set seed 42");
    s.run("gen double skip = runiform() in 1/7");
    s.run("local st = c(rngstate)");
    s.run("gen double a1 = rnormal()");
    s.run("set rngstate `st'");
    s.run("gen double a2 = rnormal()");
    assert_eq!(s.nums("a1"), s.nums("a2"));
    // other generators: reproducible, uniform, different from mt64
    for kind in ["kiss32", "mt64s"] {
        s.run(&format!("set rng {kind}"));
        assert_eq!(s.run("display c(rng_current)").trim(), kind);
        s.run("set seed 42");
        s.run(&format!("gen double {kind}_1 = runiform()"));
        s.run("set seed 42");
        s.run(&format!("gen double {kind}_2 = runiform()"));
        assert_eq!(s.nums(&format!("{kind}_1")), s.nums(&format!("{kind}_2")));
        assert_ne!(s.nums(&format!("{kind}_1")), s.nums("u0"));
        s.run(&format!("quietly summarize {kind}_1"));
        assert!((s.scalar("r(mean)") - 0.5).abs() < 0.05 && s.scalar("r(min)") >= 0.0 && s.scalar("r(max)") < 1.0);
        s.run("local st = c(rngstate)");
        s.run(&format!("gen double {kind}_3 = runiform()"));
        s.run("set rngstate `st'");
        s.run(&format!("gen double {kind}_4 = runiform()"));
        assert_eq!(s.nums(&format!("{kind}_3")), s.nums(&format!("{kind}_4")));
    }
    // mt64s streams: each (seed, stream) pair is its own sequence
    s.run("set rng mt64s");
    s.run("set rngstream 2");
    s.run("set seed 42");
    s.run("gen double st2 = runiform()");
    s.run("set rngstream 3");
    s.run("set seed 42");
    s.run("gen double st3 = runiform()");
    s.run("set rngstream 2");
    s.run("gen double st2b = runiform()");
    assert_eq!(s.nums("st2"), s.nums("st2b"));
    assert_ne!(s.nums("st2"), s.nums("st3"));
    s.run("quietly correlate st2 st3");
    assert!(s.scalar("r(rho)").abs() < 0.15);
    assert_eq!(s.run("display c(rngstream)").trim(), "2");
    s.fails("set rngstream 0");
    s.fails("set rng lcg");
    // back to the default: the original sequence
    s.run("set rng default");
    s.run("set seed 42");
    s.run("gen double u2 = runiform()");
    assert_eq!(s.nums("u0"), s.nums("u2"));
}

// ---- rocreg ----------------------------------------------------------------------

fn roc_data(s: &mut Session) {
    s.script(
        "set obs 300
         gen d = _n > 150
         gen s = mod(_n, 2)
         gen double u = (mod(_n*37, 149) + 0.5)/149
         gen double x = invnormal((mod(_n*23, 151) + 0.5)/151)
         gen double y = 0.8*s + d*(1 + 0.4*x) + invnormal(u)*(1 + 0.25*d)",
    );
}

#[test]
fn rocreg_nonparametric_matches_closed_forms() {
    let mut s = Session::new();
    roc_data(&mut s);
    s.run("roctab d y");
    let area = s.scalar("r(area)");
    let out = s.run("rocreg d y, tiecorrected breps(50) bseed(7) roc(.25) invroc(.5) pauc(.4) auc");
    assert!(out.contains("Bootstrap") && out.contains("(BC)"), "{out}");
    assert_close(s.scalar("_b[y:AUC]"), area, 1e-10);
    // placement values: ROC(t) = share of cases above the (1-t) control quantile
    let roc = s.scalar("_b[y:roc_1]");
    let pauc = s.scalar("_b[y:pauc_1]");
    let inv = s.scalar("_b[y:invroc_1]");
    assert!(roc > 0.25 && roc < 1.0 && pauc > 0.0 && pauc < 0.4 && inv > 0.0 && inv < 0.5);
    s.run("preserve");
    s.run("gsort -y");
    s.run("gen double pv = sum(d == 0)/150");
    s.run("quietly count if d == 1 & pv <= .25");
    let share = s.scalar("r(N)") / 150.0;
    s.run("quietly summarize pv if d == 1");
    let auc_pv = 1.0 - s.scalar("r(mean)");
    s.run("gen double part = max(0, .4 - pv)");
    s.run("quietly summarize part if d == 1");
    let pauc_pv = s.scalar("r(mean)");
    s.run("restore");
    assert_close(roc, share, 1e-12);
    assert_close(pauc, pauc_pv, 1e-12);
    assert_close(area, auc_pv, 1e-12);
    // the bootstrap is reproducible under bseed()
    let se = s.scalar("_se[y:AUC]");
    s.run("quietly rocreg d y, tiecorrected breps(50) bseed(7) roc(.25) invroc(.5) pauc(.4) auc");
    assert_close(s.scalar("_se[y:AUC]"), se, 0.0);
    assert!(se > 0.005 && se < 0.1);
    assert_eq!(s.run("display e(N_reps)").trim(), "50");
    // strata adjustment: the case-weighted average of within-stratum areas
    s.run("quietly rocreg d y, ctrlcov(s) ctrlmodel(strata) tiecorrected nobootstrap");
    let adj = s.scalar("_b[y:AUC]");
    let mut w = 0.0;
    for k in 0..2 {
        s.run(&format!("quietly roctab d y if s == {k}"));
        let a = s.scalar("r(area)");
        s.run(&format!("quietly count if d == 1 & s == {k}"));
        w += a * s.scalar("r(N)") / 150.0;
    }
    assert_close(adj, w, 1e-10);
    s.run("rocregplot");
}

#[test]
fn rocreg_parametric_ml_closed_form() {
    let mut s = Session::new();
    roc_data(&mut s);
    // no covariates: ML means and standard deviations (divisor n)
    s.run("quietly rocreg d y, probit ml");
    let (a, b, auc) = (s.scalar("_b[y:i_cons]"), s.scalar("_b[y:s_cons]"), s.scalar("_b[y:AUC]"));
    s.run("quietly summarize y if d == 0");
    let (m0, s0) = (s.scalar("r(mean)"), s.scalar("r(sd)") * ((s.scalar("r(N)") - 1.0) / s.scalar("r(N)")).sqrt());
    s.run("quietly summarize y if d == 1");
    let (m1, s1) = (s.scalar("r(mean)"), s.scalar("r(sd)") * ((s.scalar("r(N)") - 1.0) / s.scalar("r(N)")).sqrt());
    assert_close(a, (m1 - m0) / s1, 1e-7);
    assert_close(b, s0 / s1, 1e-7);
    assert_close(auc, s.scalar(&format!("normal({a}/sqrt(1+{b}^2))")), 1e-10);
    assert!(s.scalar("_se[y:i_cons]") > 0.0 && s.scalar("_se[y:AUC]") > 0.0);
    // covariates: the generating model is recovered approximately
    s.run("quietly rocreg d y, probit ml ctrlcov(s) roccov(x)");
    assert_close(s.scalar("_b[y:i_cons]"), 1.0 / 1.25, 0.2);
    assert_close(s.scalar("_b[y:x]"), 0.4 / 1.25, 0.15);
    assert_close(s.scalar("_b[y:s_cons]"), 1.0 / 1.25, 0.15);
    // GLM probit model agrees with ML on binormal data
    s.run("quietly rocreg d y, probit ctrlcov(s) roccov(x) breps(20) bseed(1)");
    assert_close(s.scalar("_b[y:x]"), 0.4 / 1.25, 0.2);
    assert_close(s.scalar("_b[y:s_cons]"), 1.0 / 1.25, 0.25);
    assert!(s.scalar("_se[y:x]") > 0.0);
    s.run("rocregplot, at(0)");
    s.fails("rocreg d y, roccov(x)");
}

// ---- demandsys -----------------------------------------------------------------------

/// Shares from a quadratic AIDS with known parameters (lambda = 0: AIDS),
/// plus small errors.
fn demand_data(s: &mut Session, lam: f64, noise: f64) {
    s.script(&format!(
        "set obs 400
         forvalues j = 1/3 {{
             gen double lp`j' = 0.3*invnormal((mod(_n*(7+4*`j'), 397) + 0.5)/397)
             gen double p`j' = exp(lp`j')
         }}
         gen double lm = 1 + 0.5*invnormal((mod(_n*31, 389) + 0.5)/389)
         gen double m = exp(lm)
         gen double lna = .4*lp1 + .35*lp2 + .25*lp3 + 0.5*(.1*lp1^2 + .1*lp2^2 + .08*lp3^2 - .12*lp1*lp2 - .08*lp1*lp3 - .08*lp2*lp3)
         gen double r = lm - lna
         gen double bp = exp(.1*lp1 - .05*lp2 - .05*lp3)
         gen double e1 = {noise}*invnormal((mod(_n*53, 401) + 0.5)/401)
         gen double e2 = {noise}*invnormal((mod(_n*59, 409) + 0.5)/409)
         gen double w1 = .4 + .1*lp1 - .06*lp2 - .04*lp3 + .1*r + 2*{lam}/bp*r^2 + e1
         gen double w2 = .35 - .06*lp1 + .1*lp2 - .04*lp3 - .05*r - {lam}/bp*r^2 + e2
         gen double w3 = 1 - w1 - w2"
    ));
}

#[test]
fn demandsys_aids_recovers_parameters_and_elasticities() {
    let mut s = Session::new();
    demand_data(&mut s, 0.0, 1e-4);
    let out = s.run("demandsys aids w1 w2 w3, prices(p1 p2 p3) expenditure(m)");
    assert!(out.contains("Almost ideal") && out.contains("gamma"), "{out}");
    for (n, v) in [("alpha:w1", 0.4), ("alpha:w3", 0.25), ("beta:w1", 0.1), ("beta:w3", -0.05), ("gamma:w1_p1", 0.1), ("gamma:w1_p2", -0.06), ("gamma:w2_p3", -0.04), ("gamma:w3_p3", 0.08)] {
        assert_close(s.scalar(&format!("_b[{n}]")), v, 2e-4);
    }
    // estat elasticities against the closed-form AIDS expressions
    let mut lp = [0.0; 3];
    for j in 0..3 {
        s.run(&format!("quietly summarize lp{}", j + 1));
        lp[j] = s.scalar("r(mean)");
    }
    s.run("quietly summarize lm");
    let lm = s.scalar("r(mean)");
    fn s_b(s: &mut Session, n: &str) -> f64 {
        s.scalar(&format!("_b[{n}]"))
    }
    let w = ["w1", "w2", "w3"];
    let p = ["p1", "p2", "p3"];
    let mut alpha = [0.0; 3];
    let mut beta = [0.0; 3];
    let mut g = [[0.0; 3]; 3];
    for i in 0..3 {
        alpha[i] = s_b(&mut s, &format!("alpha:{}", w[i]));
        beta[i] = s_b(&mut s, &format!("beta:{}", w[i]));
        for j in i..3 {
            g[i][j] = s_b(&mut s, &format!("gamma:{}_{}", w[i], p[j]));
            g[j][i] = g[i][j];
        }
    }
    let lna: f64 = (0..3).map(|k| alpha[k] * lp[k]).sum::<f64>() + 0.5 * (0..3).flat_map(|k| (0..3).map(move |j| (k, j))).map(|(k, j)| g[k][j] * lp[k] * lp[j]).sum::<f64>();
    s.run("estat elasticities");
    let share: Vec<f64> = (0..3).map(|i| alpha[i] + (0..3).map(|j| g[i][j] * lp[j]).sum::<f64>() + beta[i] * (lm - lna)).collect();
    for i in 0..3 {
        assert_close(s.scalar(&format!("el(r(shares),1,{})", i + 1)), share[i], 1e-9);
        assert_close(s.scalar(&format!("el(r(expenditure),1,{})", i + 1)), 1.0 + beta[i] / share[i], 1e-6);
        for j in 0..3 {
            let d = if i == j { 1.0 } else { 0.0 };
            let un = -d + (g[i][j] - beta[i] * (alpha[j] + (0..3).map(|k| g[j][k] * lp[k]).sum::<f64>())) / share[i];
            assert_close(s.scalar(&format!("el(r(uncompensated),{},{})", i + 1, j + 1)), un, 1e-6);
            assert_close(s.scalar(&format!("el(r(compensated),{},{})", i + 1, j + 1)), un + (1.0 + beta[i] / share[i]) * share[j], 1e-6);
        }
    }
    // predicted shares
    s.run("predict double wh2, equation(w2)");
    s.run("gen double dd = abs(wh2 - (w2 - e2))");
    s.run("quietly summarize dd");
    assert!(s.scalar("r(max)") < 1e-3);
}

#[test]
fn demandsys_quaids_translog_and_aggregation_identities() {
    let mut s = Session::new();
    demand_data(&mut s, 0.01, 2e-3);
    s.run("quietly demandsys quaids w1 w2 w3, prices(p1 p2 p3) expenditure(m)");
    assert_close(s.scalar("_b[lambda:w1]"), 0.02, 3e-3);
    assert_close(s.scalar("_b[lambda:w2]"), -0.01, 3e-3);
    assert_close(s.scalar("_b[lambda:w1] + _b[lambda:w2] + _b[lambda:w3]"), 0.0, 1e-12);
    assert_close(s.scalar("_b[beta:w1]"), 0.1, 5e-3);
    let ll_q = s.scalar("e(ll)");
    s.run("quietly demandsys aids w1 w2 w3, prices(p1 p2 p3) expenditure(m)");
    assert!(ll_q > s.scalar("e(ll)"));
    for model in ["quaids", "translog", "aids"] {
        s.run(&format!("quietly demandsys {model} w1 w2 w3, prices(p1 p2 p3) expenditure(m)"));
        s.run("quietly estat elasticities");
        let w: Vec<f64> = (1..=3).map(|i| s.scalar(&format!("el(r(shares),1,{i})"))).collect();
        assert_close(w.iter().sum::<f64>(), 1.0, 1e-10);
        let eta: Vec<f64> = (1..=3).map(|i| s.scalar(&format!("el(r(expenditure),1,{i})"))).collect();
        // Engel aggregation
        assert_close((0..3).map(|i| w[i] * eta[i]).sum::<f64>(), 1.0, 1e-6);
        for j in 0..3 {
            // Cournot aggregation
            let c: f64 = (0..3).map(|i| w[i] * s.scalar(&format!("el(r(uncompensated),{},{})", i + 1, j + 1))).sum();
            assert_close(c + w[j], 0.0, 1e-6);
        }
        for i in 0..3 {
            // Slutsky symmetry (imposed) and, for AIDS/QUAIDS, homogeneity
            for j in 0..3 {
                let a = w[i] * s.scalar(&format!("el(r(compensated),{},{})", i + 1, j + 1));
                let b = w[j] * s.scalar(&format!("el(r(compensated),{},{})", j + 1, i + 1));
                assert_close(a, b, 1e-6);
            }
            let h: f64 = (1..=3).map(|j| s.scalar(&format!("el(r(uncompensated),{},{j})", i + 1))).sum::<f64>() + eta[i];
            assert_close(h, 0.0, 1e-6);
        }
    }
    s.fails("demandsys linear w1 w2 w3, prices(p1 p2 p3) expenditure(m)");
    s.fails("demandsys aids w1 w2 w3, prices(p1 p2) expenditure(m)");
}

// ---- programmer's tools ------------------------------------------------------------

#[test]
fn robust_programmers_command_matches_regress_vce() {
    let mut s = Session::new();
    data(&mut s, 70);
    s.run("replace y = y * (1 + abs(x))");
    s.run("quietly regress y x z");
    s.run("predict double e, residuals");
    s.run("matrix accum XX = x z");
    s.run("matrix D = invsym(XX)");
    s.run("_robust e, variance(D) minus(3)");
    s.run("matrix D2 = invsym(XX)");
    s.run("_robust e, variance(D2) minus(3) cluster(g)");
    s.run("quietly regress y x z, vce(robust)");
    for (i, j) in [(1, 1), (2, 1), (2, 2), (3, 3), (3, 1)] {
        assert_close(s.scalar(&format!("el(D,{i},{j})")), s.scalar(&format!("el(e(V),{i},{j})")), 1e-10);
    }
    s.run("quietly regress y x z, vce(cluster g)");
    for (i, j) in [(1, 1), (2, 1), (3, 3)] {
        assert_close(s.scalar(&format!("el(D2,{i},{j})")), s.scalar(&format!("el(e(V),{i},{j})")), 1e-10);
    }
    // without variance(): e(V) itself is replaced (bread = e(V)/s^2 here)
    s.run("quietly regress y x z");
    s.run("scalar s2 = e(rmse)^2");
    s.run("matrix V0 = e(V) / s2");
    s.run("ereturn repost V = V0");
    s.run("_robust e, minus(3)");
    assert_eq!(s.run("display e(vcetype)").trim(), "Robust");
    s.run("matrix V1 = e(V)");
    s.run("quietly regress y x z, vce(robust)");
    assert_close(s.scalar("el(V1,1,1)"), s.scalar("el(e(V),1,1)"), 1e-10);
    s.fails("_robust e x");
}

#[test]
fn estat_dispatches_to_estat_cmd_and_nopreserve() {
    let mut s = Session::new();
    data(&mut s, 40);
    s.script(
        "program myreg, eclass
             syntax varlist
             quietly regress `varlist'
             ereturn local cmd \"myreg\"
             ereturn local estat_cmd \"myreg_estat\"
         end
         program myreg_estat, rclass
             gettoken sub rest : 0, parse(\" ,\")
             if \"`sub'\" == \"slope\" {
                 return scalar slope = _b[x]
             }
             else {
                 estat_default `0'
             }
         end",
    );
    s.run("myreg y x");
    s.run("estat slope");
    let b = s.scalar("_b[x]");
    assert_close(s.scalar("r(slope)"), b, 0.0);
    // standard subcommands go to the built-in estat
    let out = s.run("estat vce");
    assert!(out.contains("Covariance"), "{out}");
    s.run("estat summarize");
    // data preserved by a program are restored when it ends; with
    // nopreserve the caller keeps the only copy
    s.script(
        "program shrink
             syntax [, NOPRESERVE]
             preserve
             keep if _n <= 5
             scalar inside = _N
         end",
    );
    s.run("shrink");
    assert_eq!(s.scalar("inside"), 5.0);
    assert_eq!(s.n(), 40);
    s.run("preserve");
    s.run("shrink, nopreserve");
    assert_eq!(s.n(), 5);
    s.run("restore");
    assert_eq!(s.n(), 40);
}

// ---- information commands ------------------------------------------------------------

#[test]
fn about_search_help_query() {
    let mut s = Session::new();
    let out = s.run("about");
    assert!(out.contains("Ferrostat") && out.contains("License"), "{out}");
    let out = s.run("search roc");
    assert!(out.contains("rocreg") && out.contains("roctab") && out.contains("offline"), "{out}");
    let out = s.run("help reg");
    assert!(out.contains("regress") && out.contains("built-in"), "{out}");
    let out = s.run("help nosuchthing");
    assert!(out.contains("not found"), "{out}");
    s.run("set cformat %8.3f");
    let out = s.run("query");
    assert!(out.contains("cformat") && out.contains("%8.3f"), "{out}");
}

// ---- import sas ----------------------------------------------------------------------

/// A minimal SAS7BDAT file (32-bit, little-endian, uncompressed, one mixed
/// page): numeric x, character name, numeric d with a date format.
fn sas_file() -> Vec<u8> {
    let (header, page_size) = (1024usize, 4096usize);
    let mut h = vec![0u8; header];
    h[12..16].copy_from_slice(&[0xc2, 0xea, 0x81, 0x60]);
    h[32] = 0x22;
    h[35] = 0x22;
    h[37] = 1;
    h[70] = 20;
    h[196..200].copy_from_slice(&(header as u32).to_le_bytes());
    h[200..204].copy_from_slice(&(page_size as u32).to_le_bytes());
    h[204..208].copy_from_slice(&1u32.to_le_bytes());
    // column text blob
    let mut blob = vec![0u8; 4];
    let mut refs = std::collections::HashMap::new();
    for t in ["x", "name", "d", "DATE", "Day"] {
        refs.insert(t, (blob.len() as u16, t.len() as u16));
        blob.extend_from_slice(t.as_bytes());
        while blob.len() % 4 != 0 {
            blob.push(b' ');
        }
    }
    let tref = |t: Option<&str>| -> Vec<u8> {
        match t {
            Some(t) => {
                let (o, l) = refs[t];
                [0u16, o, l].iter().flat_map(|v| v.to_le_bytes()).collect()
            }
            None => vec![0; 6],
        }
    };
    let (ncol, row_len, nrow) = (3usize, 24usize, 3usize);
    let mut subs: Vec<Vec<u8>> = Vec::new();
    let mut rs = vec![0u8; 480];
    rs[0..4].copy_from_slice(&0xF7F7F7F7u32.to_le_bytes());
    rs[20..24].copy_from_slice(&(row_len as u32).to_le_bytes());
    rs[24..28].copy_from_slice(&(nrow as u32).to_le_bytes());
    rs[60..64].copy_from_slice(&(nrow as u32).to_le_bytes());
    subs.push(rs);
    let mut tx = 0xFFFFFFFDu32.to_le_bytes().to_vec();
    tx.extend_from_slice(&blob);
    subs.push(tx);
    let mut cn = vec![0u8; 20 + 8 * ncol];
    cn[0..4].copy_from_slice(&0xFFFFFFFFu32.to_le_bytes());
    for (i, n) in ["x", "name", "d"].iter().enumerate() {
        cn[12 + 8 * i..18 + 8 * i].copy_from_slice(&tref(Some(n)));
    }
    subs.push(cn);
    let mut ca = vec![0u8; 20 + 12 * ncol];
    ca[0..4].copy_from_slice(&0xFFFFFFFCu32.to_le_bytes());
    for (i, (off, w, t)) in [(0u32, 8u32, 1u8), (8, 8, 2), (16, 8, 1)].iter().enumerate() {
        let q = 12 + 12 * i;
        ca[q..q + 4].copy_from_slice(&off.to_le_bytes());
        ca[q + 4..q + 8].copy_from_slice(&w.to_le_bytes());
        ca[q + 10] = *t;
    }
    subs.push(ca);
    for (f, l) in [(None, None), (None, None), (Some("DATE"), Some("Day"))] {
        let mut v = vec![0u8; 52];
        v[0..4].copy_from_slice(&0xFFFFFBFEu32.to_le_bytes());
        v[34..40].copy_from_slice(&tref(f));
        v[40..46].copy_from_slice(&tref(l));
        subs.push(v);
    }
    let mut page = vec![0u8; page_size];
    page[16..18].copy_from_slice(&0x0200u16.to_le_bytes());
    page[18..20].copy_from_slice(&((subs.len() + nrow) as u16).to_le_bytes());
    page[20..22].copy_from_slice(&(subs.len() as u16).to_le_bytes());
    let mut end = page_size;
    for (k, sh) in subs.iter().enumerate() {
        end -= sh.len();
        end -= end % 8;
        page[end..end + sh.len()].copy_from_slice(sh);
        let p = 24 + 12 * k;
        page[p..p + 4].copy_from_slice(&(end as u32).to_le_bytes());
        page[p + 4..p + 8].copy_from_slice(&(sh.len() as u32).to_le_bytes());
    }
    let miss = 0xFFFF_D100_0000_0000u64.to_le_bytes();
    let num = |v: Option<f64>| -> [u8; 8] { v.map_or(miss, |x| x.to_le_bytes()) };
    let mut at = 24 + 12 * subs.len();
    if at % 8 == 4 {
        at += 4;
    }
    for (x, n, d) in [(Some(1.5), "abc", Some(21915.0)), (None, "hello", None), (Some(-2.0), "z", Some(0.0))] {
        let mut row = num(x).to_vec();
        let mut nm = n.as_bytes().to_vec();
        nm.resize(8, b' ');
        row.extend(nm);
        row.extend(num(d));
        page[at..at + row_len].copy_from_slice(&row);
        at += row_len;
    }
    h.extend(page);
    h
}

#[test]
fn import_sas_reads_a_synthetic_file() {
    let mut s = Session::new();
    let p = temp_dir("t57_sas").join("probe.sas7bdat");
    std::fs::write(&p, sas_file()).unwrap();
    s.run(&format!("import sas using \"{}\", clear", p.display()));
    assert_eq!(s.n(), 3);
    assert_eq!(s.varnames(), vec!["x", "name", "d"]);
    let x = s.nums("x");
    assert_eq!(x[0], 1.5);
    assert_eq!(x[2], -2.0);
    assert!(s.scalar("missing(x[2])") == 1.0);
    assert_eq!(s.strs("name"), vec!["abc", "hello", "z"]);
    assert_eq!(s.scalar("d[1] == td(01jan2020)"), 1.0);
    assert!(s.run("describe").contains("Day"));
}

#[test]
fn demandsys_demographics_and_rotterdam() {
    // AIDS with a demographic shift of the intercepts
    let mut s = Session::new();
    demand_data(&mut s, 0.0, 1e-4);
    s.run("gen double kids = mod(_n, 4)");
    // α_i(z) = α_i + η_i kids also enters ln a(p), hence the -β_i Σ η_k kids ln p_k terms
    s.run("replace w1 = w1 + .02*kids - .1*(.02*kids*lp1 - .01*kids*lp2 - .01*kids*lp3)");
    s.run("replace w2 = w2 - .01*kids + .05*(.02*kids*lp1 - .01*kids*lp2 - .01*kids*lp3)");
    s.run("replace w3 = 1 - w1 - w2");
    s.run("quietly demandsys aids w1 w2 w3, prices(p1 p2 p3) expenditure(m) demographics(kids)");
    assert_close(s.scalar("_b[eta:w1_kids]"), 0.02, 2e-4);
    assert_close(s.scalar("_b[eta:w2_kids]"), -0.01, 2e-4);
    assert_close(s.scalar("_b[eta:w3_kids]"), -0.01, 2e-4);
    assert_close(s.scalar("_b[gamma:w1_p2]"), -0.06, 2e-4);
    s.run("quietly estat elasticities");
    // Rotterdam on a time series generated by an AIDS with small steps
    let mut s = Session::new();
    s.script(
        "set obs 300
         gen t = _n
         tsset t
         forvalues j = 1/3 {
             gen double lp`j' = sum(0.02*invnormal((mod(_n*(11+6*`j'), 293) + 0.5)/293))
             gen double p`j' = exp(lp`j')
         }
         gen double lm = 1 + sum(0.02*invnormal((mod(_n*37, 307) + 0.5)/307))
         gen double m = exp(lm)
         gen double lna = .4*lp1 + .35*lp2 + .25*lp3 + 0.5*(.1*lp1^2 + .1*lp2^2 + .08*lp3^2 - .12*lp1*lp2 - .08*lp1*lp3 - .08*lp2*lp3)
         gen double w1 = .4 + .1*lp1 - .06*lp2 - .04*lp3 + .1*(lm - lna)
         gen double w2 = .35 - .06*lp1 + .1*lp2 - .04*lp3 - .05*(lm - lna)
         gen double w3 = 1 - w1 - w2",
    );
    let out = s.run("demandsys rotterdam w1 w2 w3, prices(p1 p2 p3) expenditure(m)");
    assert!(out.contains("Rotterdam") && out.contains("theta"), "{out}");
    assert_eq!(s.scalar("e(N)"), 299.0);
    assert_close(s.scalar("_b[theta:w1] + _b[theta:w2] + _b[theta:w3]"), 1.0, 1e-12);
    // marginal budget shares of the AIDS: w_i + β_i
    for (i, b) in [(1, 0.1), (2, -0.05), (3, -0.05)] {
        s.run(&format!("quietly summarize w{i}"));
        let w = s.scalar("r(mean)");
        assert_close(s.scalar(&format!("_b[theta:w{i}]")), w + b, 0.02);
    }
    s.run("quietly estat elasticities");
    let w: Vec<f64> = (1..=3).map(|i| s.scalar(&format!("el(r(shares),1,{i})"))).collect();
    let eta: Vec<f64> = (1..=3).map(|i| s.scalar(&format!("el(r(expenditure),1,{i})"))).collect();
    assert_close((0..3).map(|i| w[i] * eta[i]).sum::<f64>(), 1.0, 1e-10);
    for i in 0..3 {
        let h: f64 = (1..=3).map(|j| s.scalar(&format!("el(r(uncompensated),{},{j})", i + 1))).sum::<f64>() + eta[i];
        assert_close(h, 0.0, 1e-10);
    }
    s.run("quietly demandsys rotterdam w1 w2 w3, prices(p1 p2 p3) expenditure(m) noconstant");
    assert!(s.run("matrix list e(b)").contains("theta"));
    let mut s2 = Session::new();
    demand_data(&mut s2, 0.0, 1e-3);
    s2.fails("demandsys rotterdam w1 w2 w3, prices(p1 p2 p3) expenditure(m)");
}
