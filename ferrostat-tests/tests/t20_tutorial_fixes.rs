//! Fixes found by running public tutorials: prefix and macro abbreviations,
//! display separators, user egen functions, `syntax =exp`, xi interactions,
//! equation-qualified coefficients and missing model tests.

use ferrostat_tests::*;

#[test]
fn n_is_noisily_and_runs_under_capture() {
    let mut s = Session::with_fixture();
    s.run("gen v = 1");
    s.run("cap n drop v");
    assert!(!s.has_var("v"));
    let out = s.run("cap n drop nosuchvar");
    assert!(out.contains("not found"), "{out}");
    assert_eq!(s.scalar("_rc"), 111.0);
    assert_eq!(s.run("n di 3").trim(), "3");
}

#[test]
fn macro_shift_and_mac_abbreviation() {
    let mut s = Session::new();
    s.script("local 1 a\nlocal 2 b\nlocal 3 c");
    s.run("mac shift");
    assert_eq!(s.run("di \"`1'`2'`3'\"").trim(), "bc");
    s.run("macro shift 2");
    assert_eq!(s.run("di \"[`1']\"").trim(), "[]");
}

#[test]
fn display_comma_separators() {
    let mut s = Session::new();
    assert_eq!(s.run("di \"a\", \"b\",, \"c\" , 1+1, max(1,2)"), "a bc 2 2");
    s.run("tempname t");
    s.run("scalar `t' = 3");
    assert_eq!(s.run("display \"t = \", `t'"), "t =  3");
}

#[test]
fn list_without_data_is_empty() {
    let mut s = Session::new();
    assert_eq!(s.run("list in 1/3"), "");
    assert_eq!(s.run("list"), "");
}

#[test]
fn user_egen_function_via_g_program() {
    let mut s = Session::with_fixture();
    s.script(
        "program define _gtwice\nsyntax newvarname=/exp [if] [in] [, Add(real 0)]\ngen `typlist' `varlist' = 2*(`exp') + `add' `if' `in'\nend",
    );
    s.run("egen double w = twice(x), add(1)");
    let x = s.nums("x");
    let w = s.nums("w");
    for (a, b) in x.iter().zip(&w) {
        assert_close(*b, 2.0 * a + 1.0, 1e-12);
    }
    assert_eq!(s.run("di \"`: type w'\"").trim(), "double");
    s.fails("egen v = nosuchfunction(x)");
}

#[test]
fn syntax_exp_and_typlist() {
    let mut s = Session::with_fixture();
    s.script("program t\nsyntax newvarname [=exp]\ndi \"[`typlist'][`varlist'][`exp']\"\nend");
    assert_eq!(s.run("t long k = x + 1"), "[long][k][= x + 1]");
    assert_eq!(s.run("t k"), "[float][k][]");
    s.script("program t2\nsyntax newvarname =/exp\ndi \"[`exp']\"\nend");
    assert_eq!(s.run("t2 k = (x)"), "[(x)]");
    s.fails("t2 k");
}

#[test]
fn xi_star_and_bar_interactions() {
    let mut s = Session::with_fixture();
    let fv = s.run("regress y i.group##c.x");
    let b_fv = s.scalar("_b[2.group#c.x]");
    let xi = s.run("xi: regress y i.group*x");
    assert_close(s.scalar("_b[2.group#c.x]"), b_fv, 1e-10);
    assert_eq!(stat(&fv, "R-squared"), stat(&xi, "R-squared"));
    s.run("xi: regress y i.group|x");
    assert!(s.try_run("di _b[x]").is_err() || s.scalar("_b[x]") == 0.0);
}

#[test]
fn clustered_f_missing_with_more_coefficients_than_clusters() {
    let mut s = Session::with_fixture();
    // 3 clusters, 4 slopes: the overall F test cannot be computed
    let out = s.run("regress y x z cnt d, vce(cluster group)");
    assert!(out.contains("F(2, 2)") && out.contains("=          ."), "{out}");
    let out = s.run("regress y x, vce(cluster group)");
    assert!(!out.contains("=          ."), "{out}");
}

#[test]
fn equation_qualified_coefficients_after_mlogit() {
    let mut s = Session::with_fixture();
    s.script("label define g 1 one 2 two 3 three\nlabel values group g");
    let out = s.run("mlogit group x, base(1)");
    assert!(out.contains("two          |") && out.contains("Note: one is the base outcome."), "{out}");
    let two = s.scalar("_b[two:x]");
    let three = s.scalar("_b[three:x]");
    assert_ne!(two, three);
    let lin = s.run("lincom [two]x - [three]x");
    assert_close(row(&lin, "(1)").unwrap()[0], two - three, 1e-6);
    let t = s.run("test [two]x = [three]x");
    assert!(t.contains("chi2(  1)"), "{t}");
}

#[test]
fn factor_levels_show_value_labels() {
    let mut s = Session::with_fixture();
    let out = s.run("regress y i.d");
    assert!(out.contains("         yes |"), "{out}");
    s.run("set fvlabel off");
    let out = s.run("regress y i.d");
    assert!(out.contains("           1 |") && !out.contains("yes |"), "{out}");
}
