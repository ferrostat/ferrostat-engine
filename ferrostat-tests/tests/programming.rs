//! The programming language: macros, loops, conditionals, programs,
//! prefixes, stored results, matrices.

use ferrostat_tests::*;

#[test]
fn local_and_global_macros() {
    let mut s = Session::new();
    s.run("local a hello");
    s.run("global b world");
    assert_eq!(s.run("display \"`a' $b\"").trim(), "hello world");
    s.run("local n = 2 + 3");
    assert_eq!(s.run("display `n' * 2").trim(), "10");
    s.run("local s : word count a b c");
    assert_eq!(s.run("display `s'").trim(), "3");
}

#[test]
fn nested_macro_names() {
    let mut s = Session::new();
    s.script("local i 2\nlocal v2 found");
    assert_eq!(s.run("display \"`v`i''\"").trim(), "found");
}

#[test]
fn forvalues_and_foreach() {
    let mut s = Session::new();
    let out = s.script("local t 0\nforvalues i = 1/4 {\nlocal t = `t' + `i'\n}\ndisplay `t'");
    assert_eq!(out.trim(), "10");
    let out = s.script("local acc\nforeach w in a b c {\nlocal acc `acc'`w'\n}\ndisplay \"`acc'\"");
    assert_eq!(out.trim(), "abc");
    let out = s.script("foreach n of numlist 1(2)5 {\ndisplay `n'\n}");
    assert_eq!(out.split_whitespace().collect::<Vec<_>>(), ["1", "3", "5"]);
}

#[test]
fn foreach_of_varlist() {
    let mut s = Session::with_fixture();
    let out = s.script("foreach v of varlist x-z {\ndisplay \"`v'\"\n}");
    assert_eq!(out.split_whitespace().collect::<Vec<_>>(), ["x", "y", "z"]);
}

#[test]
fn while_loop_and_if_else() {
    let mut s = Session::new();
    let out = s.script("local i 0\nwhile `i' < 3 {\nlocal ++i\n}\nif `i' == 3 {\ndisplay \"three\"\n}\nelse {\ndisplay \"other\"\n}");
    assert_eq!(out.trim(), "three");
    let out = s.script("if 0 {\ndisplay \"a\"\n}\nelse if 1 {\ndisplay \"b\"\n}\nelse {\ndisplay \"c\"\n}");
    assert_eq!(out.trim(), "b");
}

#[test]
fn continue_and_break() {
    let mut s = Session::new();
    let out = s.script("forvalues i = 1/5 {\nif `i' == 2 continue\nif `i' == 4 continue, break\ndisplay `i'\n}");
    assert_eq!(out.split_whitespace().collect::<Vec<_>>(), ["1", "3"]);
}

#[test]
fn capture_sets_rc() {
    let mut s = Session::with_fixture();
    s.run("capture summarize nosuchvar");
    assert_eq!(s.scalar("_rc"), 111.0);
    s.run("capture summarize x");
    assert_eq!(s.scalar("_rc"), 0.0);
}

#[test]
fn quietly_suppresses_output() {
    let mut s = Session::with_fixture();
    assert_eq!(s.run("quietly summarize x"), "");
    assert_eq!(s.scalar("r(N)"), 12.0);
}

#[test]
fn programs_with_args_and_return() {
    let mut s = Session::new();
    s.script("program define addup, rclass\nargs a b\nreturn scalar sum = `a' + `b'\nend");
    s.run("addup 2 5");
    assert_eq!(s.scalar("r(sum)"), 7.0);
}

#[test]
fn programs_with_syntax() {
    let mut s = Session::with_fixture();
    s.script("program define mymean, rclass\nsyntax varlist(numeric) [if] [, Scale(real 1)]\nquietly summarize `varlist' `if'\nreturn scalar m = r(mean) * `scale'\nend");
    s.run("mymean x if group == 1, scale(10)");
    assert_eq!(s.scalar("r(m)"), 25.0);
    s.fails("mymean name");
}

#[test]
fn by_prefix_groupwise_generate() {
    let mut s = Session::with_fixture();
    s.run("bysort group: generate k = _n");
    s.run("by group: generate kk = _N");
    assert_eq!(&s.nums("k")[..5], [1., 2., 3., 4., 1.]);
    assert!(s.nums("kk").iter().all(|&v| v == 4.0));
}

#[test]
fn by_prefix_reporting() {
    let mut s = Session::with_fixture();
    let out = s.run("by group, sort: summarize y");
    assert_eq!(out.matches("-> group =").count(), 3, "{out}");
}

#[test]
fn assert_command() {
    let mut s = Session::with_fixture();
    s.run("assert x > 0");
    s.fails("assert x > 5");
}

#[test]
fn scalars() {
    let mut s = Session::new();
    s.run("scalar k = 3");
    assert_eq!(s.scalar("k * 2"), 6.0);
}

#[test]
fn matrices() {
    let mut s = Session::new();
    s.run("matrix A = (1, 2 \\ 3, 4)");
    assert_eq!(s.scalar("A[2,1]"), 3.0);
    s.run("matrix B = A' * A");
    assert_eq!(s.scalar("B[1,1]"), 10.0);
    s.run("matrix C = J(2, 3, 0)");
    s.run("matrix C[1,2] = 7");
    assert_eq!(s.scalar("C[1,2]"), 7.0);
    assert_eq!(s.scalar("colsof(C)"), 3.0);
    s.run("matrix symeigen V L = B");
    assert!(s.scalar("L[1,1]") > s.scalar("L[1,2]"));
    s.run("matrix D = A[1..2, 2...]");
    assert_eq!(s.scalar("rowsof(D)"), 2.0);
    assert_eq!(s.scalar("D[2,1]"), 4.0);
}

#[test]
fn mkmat_and_svmat() {
    let mut s = Session::with_fixture();
    s.run("mkmat x y, matrix(M)");
    assert_eq!(s.scalar("rowsof(M)"), 12.0);
    s.run("clear");
    s.run("svmat double M, names(col)");
    assert_eq!(s.nums("y")[0], 2.1);
}

#[test]
fn levelsof_into_local() {
    let mut s = Session::with_fixture();
    s.run("levelsof group, local(g)");
    assert_eq!(s.run("display \"`g'\"").trim(), "1 2 3");
}

#[test]
fn tempvar_and_preserve_restore() {
    let mut s = Session::with_fixture();
    s.script("preserve\ndrop if group == 1\nrestore");
    assert_eq!(s.n(), 12);
}

#[test]
fn set_obs_and_random_numbers() {
    let mut s = Session::new();
    s.run("set obs 100");
    s.run("set seed 42");
    s.run("generate double u = runiform()");
    s.run("set seed 42");
    s.run("generate double u2 = runiform()");
    assert_eq!(s.nums("u"), s.nums("u2"));
}

#[test]
fn delimit_and_continuation_in_scripts() {
    let mut s = Session::with_fixture();
    let out = s.script("display 1 + ///\n 2");
    assert_eq!(out.trim(), "3");
}

#[test]
fn display_formats() {
    let mut s = Session::new();
    assert_eq!(s.run("display %9.2f 3.14159").trim(), "3.14");
    assert_eq!(s.run("display 1/3").trim(), ".33333333");
    assert_eq!(s.run("display \"a\" \"b\"").trim(), "ab");
}

#[test]
fn unknown_command_fails_with_199() {
    let mut s = Session::new();
    s.run("capture nosuchcommand");
    assert_eq!(s.scalar("_rc"), 199.0);
}

#[test]
fn stored_results_in_macro_quotes() {
    let mut s = Session::with_fixture();
    s.run("quietly summarize x");
    let n = s.scalar("r(N)");
    assert_eq!(s.run("display \"`r(N)'\"").trim(), format!("{n}"));
    s.run("local k `c(k)'");
    assert_eq!(s.run("display `k'").trim(), format!("{}", s.ds().n_vars()));
    assert_eq!(s.run("display \"[`r(nosuch)']\"").trim(), "[]");
    // _N is the observation count, not the _n directive
    assert_eq!(s.run("display _N").trim(), format!("{}", s.n()));
}

#[test]
fn foreach_newlist_ranges() {
    let mut s = Session::with_fixture();
    s.script("foreach v of newlist z1-z3 {\ngen `v' = 1\n}");
    assert!(s.has_var("z1") && s.has_var("z2") && s.has_var("z3"));
}

#[test]
fn star_macro_and_old_display_colors() {
    let mut s = Session::new();
    s.script("program p\nargs a b\ndisplay \"[`*'] \" wordcount(\"`*'\")\nend");
    assert_eq!(s.run("p x y").trim(), "[x y] 2");
    assert_eq!(s.run("display in red \"x\" in green \"y\"").trim(), "xy");
}
