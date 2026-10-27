//! Stored-model tables: eststo, esttab, estout, estpost, estadd.

use ferrostat_tests::*;

#[test]
fn eststo_prefix_and_auto_names() {
    let mut s = Session::with_fixture();
    s.run("eststo m1: regress y x");
    let out = s.run("eststo: regress y x z");
    assert!(out.contains("(est1 stored)"), "{out}");
    let dir = s.run("eststo dir");
    assert!(dir.contains("m1") && dir.contains("est1"), "{dir}");
    s.run("estimates restore m1");
    assert_eq!(s.scalar("e(N)"), 12.0);
    s.run("eststo drop est*");
    assert!(!s.run("eststo dir").contains("est1"));
    s.run("eststo clear");
    s.fails("estimates restore m1");
}

#[test]
fn esttab_default_layout_and_stars() {
    let mut s = Session::with_fixture();
    s.run("quietly eststo a1: regress y x");
    s.run("quietly eststo a2: regress y x z");
    let out = s.run("esttab a1 a2");
    assert!(out.contains("(1)") && out.contains("(2)"), "{out}");
    assert!(out.contains("t statistics in parentheses"), "{out}");
    assert!(out.contains("* p<0.05, ** p<0.01, *** p<0.001"), "{out}");
    assert!(out.lines().any(|l| l.starts_with('x') && l.contains("***")), "{out}");
    assert!(out.lines().any(|l| l.starts_with("N ") && l.contains("12")), "{out}");
    let out = s.run("esttab a1 a2, se star(* 0.1 ** 0.05 *** 0.01) r2 label nonumbers");
    assert!(out.contains("Standard errors in parentheses"), "{out}");
    assert!(out.contains("* p<0.1, ** p<0.05, *** p<0.01"), "{out}");
    assert!(out.contains("Observations") && out.contains("R-squared") && out.contains("Constant"), "{out}");
    assert!(!out.contains("(1)"), "{out}");
    let out = s.run("esttab a2, keep(z) not nostar");
    assert!(!out.contains("_cons") && out.contains('z') && !out.contains('*'), "{out}");
}

#[test]
fn esttab_writes_files() {
    let dir = temp_dir("esttab_files");
    let mut s = Session::with_fixture();
    s.run("quietly eststo a1: regress y x");
    let rtf = dir.join("t.rtf");
    let out = s.run(&format!("esttab a1 using \"{}\", se replace", rtf.display()));
    assert!(out.contains("output written"), "{out}");
    let text = std::fs::read_to_string(&rtf).unwrap();
    assert!(text.starts_with("{\\rtf1") && text.contains("\\row") && text.trim_end().ends_with('}'), "{text}");
    let open = text.matches('{').count() - text.matches("\\{").count();
    let close = text.matches('}').count() - text.matches("\\}").count();
    assert_eq!(open, close);
    // writing again without replace fails
    s.fails(&format!("esttab a1 using \"{}\"", rtf.display()));
    let csv = dir.join("t.csv");
    s.run(&format!("esttab a1 using \"{}\", plain replace", csv.display()));
    assert!(std::fs::read_to_string(&csv).unwrap().contains("\"x\""));
    let tex = dir.join("t.tex");
    s.run(&format!("esttab a1 using \"{}\", replace booktabs", tex.display()));
    assert!(std::fs::read_to_string(&tex).unwrap().contains("\\toprule"));
}

#[test]
fn estpost_summarize_cells() {
    let mut s = Session::with_fixture();
    s.run("eststo d0: estpost summarize x y if d == 0, detail");
    assert_eq!(s.scalar("el(e(mean),1,1)"), 29.0 / 6.0);
    assert_eq!(s.scalar("el(e(count),1,2)"), 6.0);
    s.run("eststo d1: estpost summarize x y if d == 1");
    let out = s.run("esttab d0 d1, cells(mean(fmt(2)) sd(fmt(2) par)) mtitles(\"No\" \"Yes\") label");
    assert!(out.contains("No") && out.contains("Yes"), "{out}");
    assert!(out.contains("4.83") && out.contains("(3.31)"), "{out}");
}

#[test]
fn estpost_ttest_cells() {
    let mut s = Session::with_fixture();
    s.run("eststo t: estpost ttest x y, by(d)");
    assert_eq!(s.scalar("el(e(N_1),1,1)"), 6.0);
    assert_eq!(s.scalar("el(e(mu_1),1,1)"), 29.0 / 6.0);
    assert_close(s.scalar("el(e(b),1,1)"), -10.0 / 3.0, 1e-9);
    let out = s.run("esttab t, cells(\"N_1 mu_1(fmt(2)) N_2 mu_2(fmt(2)) b(fmt(2)) p(fmt(4))\") noobs");
    assert!(out.contains("mu_1") && out.contains("4.83") && out.contains("8.17"), "{out}");
    assert!(!out.contains("\nN "), "{out}");
}

#[test]
fn estpost_tabulate_and_tabstat() {
    let mut s = Session::with_fixture();
    s.run("estpost tabulate d");
    assert_eq!(s.scalar("el(e(b),1,1)"), 6.0);
    assert_eq!(s.scalar("e(N)"), 12.0);
    let out = s.run("esttab ., cells(\"b pct(fmt(1))\") label");
    assert!(out.contains("yes") && out.contains("50.0"), "{out}");
    s.run("estpost tabulate group d");
    let out = s.run("esttab ., cells(colpct(fmt(1))) unstack noobs");
    assert!(out.contains("Total"), "{out}");
    s.run("estpost tabstat x y, by(d) statistics(mean sd)");
    assert_eq!(s.scalar("el(e(mean),1,1)"), 29.0 / 6.0);
}

#[test]
fn estadd_and_estout() {
    let mut s = Session::with_fixture();
    s.run("quietly eststo: regress y x");
    s.run("estadd local fe \"Yes\"");
    s.run("estadd scalar k2 = 1 + 1");
    let out = s.run("esttab, stats(N k2 fe)");
    assert!(out.contains("Yes") && out.contains("k2"), "{out}");
    let out = s.run("estout, cells(b(star fmt(3)) se(par))");
    assert!(out.contains("est1\n") || out.contains("\test1"), "{out}");
    assert!(out.contains('\t'), "{out}");
}

#[test]
fn ssc_install_estout_is_native() {
    let mut s = Session::new();
    s.run("ssc install estout");
    s.run("ssc install estout, replace");
}
