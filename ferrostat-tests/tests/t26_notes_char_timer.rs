//! `notes`, `char`, `clist` and `timer`.

use ferrostat_tests::*;

#[test]
fn notes_add_list_drop_and_survive_save() {
    let mut s = Session::with_fixture();
    s.run("notes: dataset note");
    s.run("notes x: first");
    s.run("notes x: second");
    let out = s.run("notes");
    assert!(out.contains("_dta:\n  1.  dataset note") && out.contains("x:\n  1.  first\n  2.  second"), "{out}");
    assert_eq!(s.run("di \"`: char x[note0]'\"").trim(), "2");
    assert_eq!(s.run("notes drop x in 1").trim(), "(1 note dropped)");
    assert!(s.run("notes x").contains("1.  second"));
    s.run("notes replace x in 1: changed");
    assert!(s.run("notes list x").contains("1.  changed"));
    let dir = temp_dir("notes");
    let f = dir.join("n.dta").display().to_string();
    s.run(&format!("save \"{f}\", replace"));
    s.run(&format!("use \"{f}\", clear"));
    assert!(s.run("notes").contains("1.  changed"));
}

#[test]
fn char_define_list_delete() {
    let mut s = Session::with_fixture();
    s.run("char x[unit] kg");
    s.run("char _dta[source] survey");
    assert_eq!(s.run("di \"`: char x[unit]'\"").trim(), "kg");
    let out = s.run("char list");
    assert!(out.contains("x[unit]:") && out.contains("_dta[source]:"), "{out}");
    s.run("char x[unit]");
    assert_eq!(s.run("di \"[`: char x[unit]']\"").trim(), "[]");
}

#[test]
fn clist_is_list_clean() {
    let mut s = Session::with_fixture();
    assert_eq!(s.run("clist id x in 1/2"), s.run("list id x in 1/2, clean"));
    // header and values line up (right-aligned numeric column)
    for opts in ["", ", noobs"] {
        let out = s.run(&format!("clist id in 1/2{opts}"));
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0].trim_end().len(), lines[1].trim_end().len(), "{out}");
    }
}

#[test]
fn timer_on_off_list() {
    let mut s = Session::with_fixture();
    s.run("timer clear");
    s.run("timer on 2");
    s.run("summarize x");
    s.run("timer off 2");
    s.run("timer on 2");
    s.run("timer off 2");
    let out = s.run("timer list");
    assert!(out.trim_start().starts_with("2:"), "{out}");
    assert_eq!(s.scalar("r(nt2)"), 2.0);
    assert!(s.scalar("r(t2)") >= 0.0);
    s.fails("timer off 2");
    s.fails("timer on 101");
}

#[test]
fn graph_generic_weight_accepts_fractions() {
    let mut s = Session::with_fixture();
    s.run("gen w = x / 3");
    s.run("twoway scatter y x [weight=w], msymbol(oh)");
    s.run("scatter y x [aw=w]");
    s.fails("summarize y [weight=w]");
}

#[test]
fn label_languages_switch_and_save() {
    let mut s = Session::with_fixture();
    s.run("label variable x \"Weight\"");
    s.run("label language es, new");
    assert_eq!(s.run("di \"[`: variable label x']\"").trim(), "[]");
    assert_eq!(s.run("di \"[`: value label d']\"").trim(), "[]");
    s.run("label variable x \"Peso\"");
    s.run("label define sino 0 no 1 si");
    s.run("label values d sino");
    s.run("label language default");
    assert_eq!(s.run("di \"`: variable label x'\"").trim(), "Weight");
    assert_eq!(s.run("di \"`: value label d'\"").trim(), "yesno");
    let dir = temp_dir("lang");
    let f = dir.join("l.dta").display().to_string();
    s.run(&format!("save \"{f}\", replace"));
    s.run(&format!("use \"{f}\", clear"));
    s.run("label language es");
    assert_eq!(s.run("di \"`: variable label x'\"").trim(), "Peso");
    assert_eq!(s.run("di \"`: value label d'\"").trim(), "sino");
    assert!(s.run("label language").contains("Currently active language: es"));
    s.run("label language spanish, rename");
    s.run("label language default, delete");
    assert!(s.run("label language").contains("only one language: spanish"));
    s.fails("label language nosuch");
}

#[test]
fn proportion_estimates_and_covariance() {
    let mut s = Session::with_fixture();
    let out = s.run("proportion d");
    assert!(out.contains("Logit") && out.contains("no |") && out.contains("yes |"), "{out}");
    let n = s.scalar("e(N)");
    let p0 = s.scalar("_b[0.d]");
    let p1 = s.scalar("_b[1.d]");
    assert_close(p0 + p1, 1.0, 1e-12);
    assert_close(s.scalar("_se[0.d]"), (p0 * p1 / (n - 1.0)).sqrt(), 1e-12);
    // the difference of complementary proportions has twice the SE
    let lin = s.run("lincom 0.d - 1.d");
    assert!(lin.contains("      t    P>|t|"), "{lin}");
    assert_close(row(&lin, "(1)").unwrap()[1], 2.0 * s.scalar("_se[0.d]"), 1e-6);
    let w = s.run("proportion group, citype(wald)");
    assert!(!w.contains("Logit"));
    s.run("proportion d, over(group)");
    assert!(s.scalar("_b[1.d@2.group]") >= 0.0);
}

#[test]
fn matrix_named_like_a_subcommand() {
    let mut s = Session::new();
    s.run("matrix A = (1,2)");
    s.run("matrix L = A");
    s.run("matrix dir = A*2");
    assert_eq!(s.scalar("L[1,2]"), 2.0);
    assert_eq!(s.scalar("dir[1,2]"), 4.0);
    assert!(s.run("matrix list L").contains("L[1,2]"));
}

#[test]
fn use_restores_tsset() {
    let mut s = Session::with_fixture();
    s.run("tsset id");
    let dir = temp_dir("tsuse");
    let f = dir.join("ts.dta").display().to_string();
    s.run(&format!("save \"{f}\", replace"));
    s.run("clear");
    s.run(&format!("use \"{f}\""));
    s.run("gen lx = L.x");
    assert!(s.nums("lx")[0].is_nan() || s.nums("lx")[0] > 1e300);
    assert_eq!(s.nums("lx")[1], s.nums("x")[0]);
    s.run("xtset group id");
    s.run(&format!("save \"{f}\", replace"));
    s.run(&format!("use \"{f}\", clear"));
    assert!(s.run("xtset").contains("group"));
}
