//! Parallel replications: results must not depend on the number of
//! processors, and the session's random-number stream advances the same way.

use ferrostat_tests::*;

fn boot(s: &mut Session, procs: usize) -> (Vec<f64>, f64) {
    s.run(&format!("set processors {procs}"));
    s.run("quietly bootstrap _b, reps(60) seed(11) nodots: regress y x z");
    let se = vec![s.scalar("_se[x]"), s.scalar("_se[z]"), s.scalar("_se[_cons]")];
    (se, s.scalar("runiform()"))
}

#[test]
fn bootstrap_same_for_any_processor_count() {
    let mut s = Session::with_fixture();
    let one = boot(&mut s, 1);
    let two = boot(&mut s, 2);
    let four = boot(&mut s, 4);
    assert_eq!(one, two);
    assert_eq!(one, four);
    assert!(one.0.iter().all(|v| *v > 0.0));
}

#[test]
fn bca_jackknife_and_permute_and_simulate_deterministic() {
    let mut s = Session::with_fixture();
    let mut run = |procs: usize| {
        s.run(&format!("set processors {procs}"));
        s.run("quietly bootstrap m=r(mean), reps(40) seed(5) bca nodots: summarize y");
        s.run("matrix bcaci = e(ci_bca)");
        let a = (s.scalar("_se[m]"), s.scalar("bcaci[1,1]"), s.scalar("bcaci[2,1]"));
        s.run("quietly permute x m=r(mean), reps(30) seed(9) nodots: summarize y if x > 2");
        s.run("matrix pm = r(p)");
        let p = s.scalar("pm[1,1]");
        (a, p)
    };
    assert_eq!(run(1), run(2));
}

#[test]
fn simulate_parallel_matches_sequential() {
    let mut s = Session::new();
    s.script("program sim1, rclass\ndrop _all\nset obs 30\ngen v = rnormal()\nsummarize v\nreturn scalar m = r(mean)\nend");
    let mut means = Vec::new();
    for procs in [1, 2] {
        s.run(&format!("set processors {procs}"));
        s.run("simulate m=r(m), reps(25) seed(3) nodots: sim1");
        assert_eq!(s.n(), 25);
        means.push(s.nums("m"));
    }
    assert_eq!(means[0], means[1]);
    assert!(means[0].windows(2).any(|w| w[0] != w[1]));
}

#[test]
fn processors_setting_is_reported() {
    let mut s = Session::new();
    s.run("set processors 1");
    assert_eq!(s.scalar("c(processors)"), 1.0);
}

#[test]
fn text_transport_matches_threads() {
    // the browser sends jobs to helper workers as text plus .dta bytes;
    // `set parallel_transport text` runs that protocol on threads
    let mut s = Session::with_fixture();
    s.script("global scale 3\nscalar shift = 0.5\nmatrix w = (1, 2 \\ 3, 4)\nprogram mystat, rclass\nsummarize y, meanonly\nreturn scalar m = r(mean) * $scale + shift + w[2,1]\nend");
    s.run("label define gl 1 one 2 two 3 three");
    s.run("label values group gl");
    let mut run = |transport: &str| {
        s.run("set processors 2");
        s.run(&format!("set parallel_transport {transport}"));
        s.run("quietly bootstrap m=r(m), reps(30) seed(4) strata(group) nodots: mystat");
        let a = (s.scalar("_b[m]"), s.scalar("_se[m]"));
        s.run("quietly bootstrap _b, reps(20) seed(4) cluster(group) idcluster(newg) nodots: regress y x");
        let b = (s.scalar("_se[x]"), s.scalar("_se[_cons]"));
        (a, b)
    };
    let threads = run("threads");
    let text = run("text");
    assert_eq!(threads, text);
    assert!(threads.0 .1 > 0.0);
}
