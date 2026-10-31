//! Regression tests for crashes and inconsistencies found in the core:
//! factor variables in `matrix accum`, date formats on out-of-range values,
//! macro expansion in single-line `else`, and one-target `gettoken`.

use ferrostat_tests::*;

fn fv_data(s: &mut Session) {
    s.script(
        "set obs 6
         gen x = mod(_n,3) + 1
         gen y = _n",
    );
}

// ---- matrix accum with factor variables ---------------------------------------

#[test]
fn matrix_accum_factor_variables() {
    let mut s = Session::new();
    fv_data(&mut s);
    let out = s.run("matrix accum A = i.x y");
    assert!(out.contains("obs=6"), "{out}");
    let l = s.run("matrix list A");
    // base level first, indicator names, constant last
    assert!(l.contains("1b.x") && l.contains("2.x") && l.contains("3.x") && l.contains("_cons"), "{l}");
    assert_eq!(s.scalar("rowsof(A)"), 5.0);
    // the base column is all zeros
    assert_eq!(s.scalar("A[1,1]"), 0.0);
    assert_eq!(s.scalar("A[5,1]"), 0.0);
    // x = 2 at _n = 1, 4; x = 3 at _n = 2, 5
    assert_eq!(s.scalar("A[2,2]"), 2.0);
    assert_eq!(s.scalar("A[3,3]"), 2.0);
    assert_eq!(s.scalar("A[4,2]"), 5.0);
    assert_eq!(s.scalar("A[4,3]"), 7.0);
    assert_eq!(s.scalar("A[4,4]"), 91.0);
    assert_eq!(s.scalar("A[5,4]"), 21.0);
    assert_eq!(s.scalar("A[5,5]"), 6.0);
    assert_eq!(s.scalar("r(N)"), 6.0);
}

#[test]
fn matrix_accum_ibn_and_interactions() {
    let mut s = Session::new();
    fv_data(&mut s);
    s.run("matrix accum B = ibn.x, noconstant");
    let l = s.run("matrix list B");
    assert!(l.contains("1.x") && !l.contains("1b.x") && !l.contains("_cons"), "{l}");
    for k in 1..=3 {
        assert_eq!(s.scalar(&format!("B[{k},{k}]")), 2.0);
    }
    s.run("matrix accum C = c.y#i.x");
    assert_eq!(s.scalar("colsof(C)"), 4.0);
    // y^2 summed over x = 2 (1, 4) and x = 3 (2, 5)
    assert_eq!(s.scalar("C[2,2]"), 17.0);
    assert_eq!(s.scalar("C[3,3]"), 29.0);
    // plain varlists are unchanged
    s.run("matrix accum D = x y");
    assert_eq!(s.scalar("D[1,1]"), 1.0 + 4.0 + 9.0 + 1.0 + 4.0 + 9.0);
    assert_eq!(s.scalar("D[3,3]"), 6.0);
}

#[test]
fn matrix_expression_errors_do_not_panic() {
    let mut s = Session::new();
    s.run("matrix A = (1,2\\3,4)");
    s.fails("matrix B = J(1e12,1e12,0)");
    s.fails("matrix B = J(-1,2,0)");
    s.fails("matrix B = I(1e15)");
    s.fails("matrix B = A[1");
    s.fails("matrix B = hadamard(A)");
    s.fails("matrix B = diag()");
    s.fails("matrix B = corr((1,2,3))");
    s.fails("matrix A]1[ = 5");
    s.fails("matrix B = inv(");
    // valid calls still work
    s.run("matrix B = hadamard(A, A)");
    assert_eq!(s.scalar("B[2,2]"), 16.0);
}

// ---- date formats on unrepresentable values ----------------------------------

#[test]
fn date_formats_out_of_range() {
    let mut s = Session::new();
    for (fmt, v) in [
        ("%td", "1e10"),
        ("%td", "-1e300"),
        ("%td", "2936550"),
        ("%td", "-679351"),
        ("%tdCCYY-NN-DD", "9e15"),
        ("%tc", "1e300"),
        ("%tcCCYY-NN-DD_HH:MM", "-1e300"),
        ("%tm", "1e300"),
        ("%tq", "-1e300"),
        ("%th", "1e20"),
        ("%tw", "1e20"),
        ("%ty", "1e20"),
    ] {
        let out = s.run(&format!("display {fmt} {v}"));
        assert_eq!(out.trim(), ".", "{fmt} {v}");
    }
    // the ends of the supported range still display
    assert!(s.run("display %td -679350").trim().starts_with("01jan"));
    assert_eq!(s.run("display %td 2936549").trim(), "31dec9999");
    assert_eq!(s.run("display %td 0").trim(), "01jan1960");
    assert_eq!(s.run("display %tm 0").trim(), "1960m1");
    // a formatted variable holding a huge value lists without panicking
    s.script(
        "set obs 2
         gen double d = 1e15 in 1
         replace d = 0 in 2
         format d %td",
    );
    let l = s.run("list d");
    assert!(l.contains("01jan1960"), "{l}");
}

#[test]
fn date_functions_out_of_range_are_missing() {
    let mut s = Session::new();
    for e in [
        "year(1e10)",
        "month(-1e300)",
        "dow(1e300)",
        "doy(1e18)",
        "mdy(1,1,1e18)",
        "mdy(1,1e18,2000)",
        "dofm(1e300)",
        "dofq(-1e300)",
        "dofh(1e20)",
        "dofw(1e20)",
        "dofy(1e20)",
        "mofd(1e20)",
        "wofd(-1e20)",
        "birthday(0, 1e18)",
    ] {
        let out = s.run(&format!("display {e}"));
        assert_eq!(out.trim(), ".", "{e}");
    }
    assert_eq!(s.scalar("year(0)"), 1960.0);
    assert_eq!(s.scalar("dow(0)"), 5.0);
    assert_eq!(s.scalar("mdy(12,31,9999)"), 2936549.0);
    // compact numeric date with a mask that does not fit 8 digits
    assert_eq!(s.run("display date(\"20000115\", \"YYMD\")").trim(), ".");
    assert_eq!(s.scalar("date(\"20000115\", \"YMD\")"), 14624.0);
}

// ---- single-line else ---------------------------------------------------------

#[test]
fn single_line_else_expands_macros() {
    let mut s = Session::new();
    s.run("local a hello");
    s.run("if 0 display \"if: `a'\"");
    let out = s.run("else display \"else: `a'\"");
    assert_eq!(out.trim(), "else: hello");
    // else if chain
    s.run("if 0 display \"no\"");
    s.run("else if 0 display \"no\"");
    let out = s.run("else display \"chain: `a'\"");
    assert_eq!(out.trim(), "chain: hello");
    // braced else and else inside a program
    let out = s.script(
        "capture program drop t58e
         program define t58e
             args v
             local w world
             if `v' display \"one `w'\"
             else display \"other `w'\"
             if `v' == 1 {
                 display \"braced `w'\"
             }
             else {
                 display \"braced else `w'\"
             }
         end
         t58e 0",
    );
    assert!(out.contains("other world") && out.contains("braced else world"), "{out}");
    let out = s.run("t58e 1");
    assert!(out.contains("one world") && out.contains("braced world"), "{out}");
}

// ---- gettoken -----------------------------------------------------------------

#[test]
fn gettoken_one_target_keeps_source() {
    let mut s = Session::new();
    s.run("local 0 one two three");
    s.run("gettoken tok : 0");
    assert_eq!(s.run("display \"`tok'|`0'\"").trim(), "one|one two three");
    s.run("gettoken tok rest : 0");
    assert_eq!(s.run("display \"`tok'|`rest'|`0'\"").trim(), "one| two three|one two three");
    // the classic consuming loop names the source as the remainder
    let out = s.script(
        "local list a b c
         local n 0
         while \"`list'\" != \"\" {
             gettoken w list : list
             local n = `n' + 1
         }
         display `n'",
    );
    assert_eq!(out.trim(), "3");
    // parse() and (global) designators
    s.run("local s \"x,y\"");
    s.run("gettoken a b : s, parse(\",\")");
    assert_eq!(s.run("display \"`a'|`b'|`s'\"").trim(), "x|,y|x,y");
    s.run("global G \"g1 g2\"");
    s.run("gettoken (global) H : (global) G");
    assert_eq!(s.run("display \"$H|$G\"").trim(), "g1|g1 g2");
    s.fails("gettoken : 0");
    s.fails("gettoken a b c : 0");
}
