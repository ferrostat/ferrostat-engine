//! Regressions found re-running the Göttingen economic-analysis course
//! do-files (wave W13): `syntax` required options and option-argument
//! checks, `[if/]`, string options without their quotes, numlist
//! constraints, `display _asis`, `r(table)[i,j]` in expressions, and the
//! messages of `replace` (type promotion), `tostring` and `codebook`.
//! All data here is synthetic.

use ferrostat_ado::interpreter::rc_of;
use ferrostat_tests::*;

fn rc(s: &mut Session, line: &str) -> i32 {
    match s.try_run(line) {
        Ok(_) => 0,
        Err(e) => rc_of(&e),
    }
}

/// Six observations: a numeric `v`, a second numeric `w` and a string `name`.
fn data(s: &mut Session) {
    s.script(
        "set obs 6
         gen v = _n * 10
         gen w = 7 - _n
         gen str5 name = \"n\" + string(_n)",
    );
}

// ---- syntax: required options and argument types --------------------------------

#[test]
fn syntax_options_outside_brackets_are_required() {
    let mut s = Session::new();
    data(&mut s);
    s.script(
        "capture program drop t62a
         program t62a
             syntax varlist [if], REplace m(numlist >10) [n(integer 5)] [k(varname numeric)]
             display \"r=`replace' m=`m' n=`n' k=`k'\"
         end",
    );
    // a required flag and a required option with an argument
    assert_eq!(rc(&mut s, "t62a v"), 198);
    assert!(s.try_run("t62a v").unwrap_err().contains("option replace required"));
    assert!(s.try_run("t62a v, replace").unwrap_err().contains("option m() required"));
    // abbreviation of the required flag, optional integer default
    let out = s.run("t62a v, re m(20 30)");
    assert_eq!(out.trim(), "r=replace m=20 30 n=5 k=");
    // integer argument checked
    assert_eq!(rc(&mut s, "t62a v, replace m(20) n(abc)"), 198);
    assert_eq!(rc(&mut s, "t62a v, replace m(20) n(2.5)"), 198);
    // numlist range constraint
    assert_eq!(rc(&mut s, "t62a v, replace m(10 20)"), 125);
    // varname numeric: string variable rejected, two variables rejected
    assert_eq!(rc(&mut s, "t62a v, replace m(20) k(name)"), 109);
    assert_eq!(rc(&mut s, "t62a v, replace m(20) k(v w)"), 103);
    let out = s.run("t62a v, replace m(20) n(10) k(w)");
    assert_eq!(out.trim(), "r=replace m=20 n=10 k=w");
}

#[test]
fn syntax_bracketed_option_list_stays_optional() {
    let mut s = Session::new();
    data(&mut s);
    s.script(
        "capture program drop t62b
         program t62b
             syntax varlist [, Detail by(varname) noLABel]
             display \"d=`detail' by=`by' l=`label'\"
         end",
    );
    assert_eq!(s.run("t62b v").trim(), "d= by= l=");
    assert_eq!(s.run("t62b v, d by(w) nolab").trim(), "d=detail by=w l=nolabel");
    // several options inside one pair of brackets, plus a required one after them
    s.script(
        "capture program drop t62b2
         program t62b2
             syntax [anything], [n(real 10) alpha(real 0.05) onesided] title(string)
             display \"n=`n' a=`alpha' o=`onesided' t=`title'\"
         end",
    );
    assert_eq!(s.run("t62b2 x, alpha(0.1) title(\"T\")").trim(), "n=10 a=.1 o= t=T");
    assert!(s.try_run("t62b2 x, alpha(0.1)").unwrap_err().contains("option title() required"));
}

#[test]
fn syntax_numlist_count_and_name_options() {
    let mut s = Session::new();
    s.script(
        "capture program drop t62c
         program t62c, rclass
             syntax, NUMbers(numlist min=3 >10) type(name max=1)
             local t 0
             foreach n of local numbers {
                 local t = `t' + `n'
             }
             return scalar total = `t'
             return local type `type'
         end",
    );
    s.run("t62c, num(100 50 23) type(sum)");
    assert_eq!(s.scalar("r(total)"), 173.0);
    // too few elements, element out of range, two names, invalid name
    assert_eq!(rc(&mut s, "t62c, num(50 23) type(sum)"), 122);
    assert_eq!(rc(&mut s, "t62c, num(1 50 23) type(sum)"), 125);
    assert_eq!(rc(&mut s, "t62c, num(11 50 23) type(a b)"), 198);
    assert_eq!(rc(&mut s, "t62c, num(11 50 23) type(3)"), 198);
}

#[test]
fn syntax_string_option_drops_quotes_unless_asis() {
    let mut s = Session::new();
    s.script(
        "capture program drop t62d
         program t62d
             syntax [, plot(string) raw(string asis) pass(passthru)]
             display `\"[`plot'] [`raw'] [`pass']\"'
         end",
    );
    let out = s.run("t62d, plot(\"Graph\") raw(\"a b\") pass(x y)");
    assert_eq!(out.trim(), "[Graph] [\"a b\"] [pass(x y)]");
    let out = s.run("t62d, plot(two words)");
    assert_eq!(out.trim(), "[two words] [] []");
}

#[test]
fn syntax_if_slash_and_marksample() {
    let mut s = Session::new();
    data(&mut s);
    s.script(
        "capture program drop t62e
         program t62e, rclass
             syntax varlist [if/] [in]
             return local cond `\"`if'\"'
             quietly count if v > 15 & `if' `in'
             return scalar n1 = r(N)
             marksample touse
             quietly count if `touse'
             return scalar n2 = r(N)
         end",
    );
    s.run("t62e v if w >= 3");
    assert_eq!(s.run("display \"`r(cond)'\"").trim(), "w >= 3");
    // v > 15 and w >= 3: observations 2, 3, 4
    assert_eq!(s.scalar("r(n1)"), 3.0);
    // w >= 3: observations 1-4
    assert_eq!(s.scalar("r(n2)"), 4.0);
    s.run("t62e v if w >= 3 in 1/2");
    assert_eq!(s.scalar("r(n1)"), 1.0);
    assert_eq!(s.scalar("r(n2)"), 2.0);
}

// ---- numlist command constraints ---------------------------------------------

#[test]
fn numlist_command_constraints() {
    let mut s = Session::new();
    assert_eq!(rc(&mut s, "numlist \"1 2 3\", min(4)"), 122);
    assert_eq!(rc(&mut s, "numlist \"1/5\", max(3)"), 123);
    assert_eq!(rc(&mut s, "numlist \"1 5 3\", ascending"), 124);
    assert_eq!(rc(&mut s, "numlist \"1 50\", range(>=1 <=10)"), 125);
    assert_eq!(rc(&mut s, "numlist \"1 2.5\", integer"), 126);
    s.run("numlist \"3 1 2\", sort");
    assert_eq!(s.run("display \"`r(numlist)'\"").trim(), "1 2 3");
}

// ---- display _asis -------------------------------------------------------------

#[test]
fn display_asis_directive() {
    let mut s = Session::new();
    let out = s.run("display _asis \"difference is \" 3*2");
    assert_eq!(out.trim(), "difference is 6");
    // after _asis, braces are printed literally
    let out = s.run("display _asis \"{x}\"");
    assert_eq!(out.trim(), "{x}");
}

// ---- stored-result matrices subscripted in expressions -----------------------------

#[test]
fn stored_matrix_subscripts_in_expressions() {
    let mut s = Session::new();
    data(&mut s);
    s.run("quietly regress v w");
    // v = 70 - 10 w exactly: coefficient -10, constant 70
    assert_close(s.scalar("r(table)[1,1]"), -10.0, 1e-9);
    assert_close(s.scalar("r(table)[1,2]"), 70.0, 1e-9);
    s.run("local n = 2");
    assert_close(s.scalar("r(table)[1,`n']"), 70.0, 1e-9);
    assert_close(s.scalar("e(b)[1,1]"), -10.0, 1e-9);
    s.run("local s = cond(r(table)[1,1] < 0, \"neg\", \"pos\")");
    assert_eq!(s.run("display \"`s'\"").trim(), "neg");
    assert_eq!(rc(&mut s, "display r(nosuchmatrix)[1,1]"), 111);
}

// ---- messages --------------------------------------------------------------------

#[test]
fn replace_promotion_names_old_type() {
    let mut s = Session::new();
    s.run("set obs 3");
    s.run("gen long k = 1000 * _n");
    let out = s.run("replace k = k / 1000 + 0.5");
    assert!(out.contains("variable k was long now double"), "{out}");
}

#[test]
fn tostring_messages() {
    let mut s = Session::new();
    s.run("set obs 3");
    s.run("gen int a = _n * 100");
    s.run("gen float b = _n");
    let out = s.run("tostring a, generate(as)");
    assert_eq!(out.trim(), "as generated as str3");
    let out = s.run("tostring b, replace");
    assert_eq!(out.trim(), "b was float now str1");
}

#[test]
fn codebook_unique_values_exclude_missing() {
    let mut s = Session::new();
    s.script(
        "set obs 10
         gen x = mod(_n, 3)
         replace x = . in 1/2
         gen y = _n / 4
         label define yn 0 \"no\" 1 \"yes\" 2 \"maybe\"
         label values x yn",
    );
    let out = s.run("codebook x");
    assert!(out.contains("Unique values: 3"), "{out}");
    assert!(out.contains("missing .: 2/10"), "{out}");
    // few values: a tabulation with the labels
    assert!(out.contains("Tabulation") && out.contains("maybe"), "{out}");
    let out = s.run("codebook y");
    assert!(out.contains("Unique values: 10"), "{out}");
    assert!(out.contains("units: .01"), "{out}");
    assert!(out.contains("Mean:") && out.contains("Percentiles"), "{out}");
}

// ---- nested global names -----------------------------------------------------------

#[test]
fn nested_global_names_in_braces() {
    let mut s = Session::new();
    s.script(
        "global x2 = 5
         global i = 2
         global j x",
    );
    assert_eq!(s.run("display \"${x$i}\"").trim(), "5");
    assert_eq!(s.run("display \"${x${i}}\"").trim(), "5");
    assert_eq!(s.run("display \"${${j}2}|$x$i\"").trim(), "5|2");
}

// ---- macro assignment strips the first and last quote ---------------------------

#[test]
fn local_strips_leading_and_trailing_quote() {
    let mut s = Session::new();
    s.run("local a \"Audi Fox\"");
    assert_eq!(s.run("display `\"`a'\"'").trim(), "Audi Fox");
    // the outer quotes are removed even though they do not pair up
    s.run("local b \"\" \"Audi Fox\"");
    assert_eq!(s.run("display `\"`b'\"'").trim(), "\" \"Audi Fox");
    s.run("local c \"one\" \"two\"");
    assert_eq!(s.run("display `\"`c'\"'").trim(), "one\" \"two");
    // compound quotes keep the inner ones
    s.run("local d `\" \"one\" \"two\" \"'");
    assert_eq!(s.run("display `\"`d'\"'").trim(), "\"one\" \"two\"");
    s.run("local e plain text");
    assert_eq!(s.run("display \"`e'\"").trim(), "plain text");
}

// ---- spmap palettes and legend options ----------------------------------------------

#[test]
fn spmap_palette_names() {
    let blues = ferrostat_cmd::spatial::palette("Blues", 5).unwrap();
    assert_eq!(blues.len(), 5);
    // light to dark
    assert!(blues[0].2 > blues[4].2 && blues[0].0 > blues[4].0);
    assert_eq!(blues[0], (0xf7, 0xfb, 0xff));
    assert_eq!(ferrostat_cmd::spatial::palette("greens", 1).unwrap().len(), 1);
    assert!(ferrostat_cmd::spatial::palette("navy", 3).is_none());
}

#[test]
fn spmap_value_labels_counts_and_title_in_legend() {
    let dir = temp_dir("t62_spmap");
    let coord = dir.join("coord.dta").display().to_string();
    let svg = dir.join("map.svg").display().to_string();
    let mut s = Session::new();
    // four unit squares, five vertices each
    s.script(
        "set obs 20
         gen _ID = ceil(_n / 5)
         gen k = mod(_n - 1, 5)
         gen _X = (_ID - 1) + inlist(k, 1, 2)
         gen _Y = inlist(k, 2, 3)
         drop k",
    );
    s.run(&format!("save \"{coord}\", replace"));
    s.script(
        "clear
         set obs 4
         gen id = _n
         gen level = cond(_n == 1, 0, cond(_n < 4, 1, 2))
         label define lev 0 \"no data\" 1 \"low\" 2 \"high\"
         label values level lev",
    );
    s.run(&format!(
        "spmap level using \"{coord}\", id(id) fcolor(Blues) clmethod(unique) legtitle(\"Share\") leglabel(lev) legcount"
    ));
    s.run(&format!("graph export \"{svg}\", replace"));
    let text = std::fs::read_to_string(&svg).unwrap();
    for want in ["Share", "no data (1)", "low (2)", "high (1)"] {
        assert!(text.contains(want), "legend entry {want:?} missing");
    }
    // legenda(off) drops the legend
    s.run(&format!("spmap level using \"{coord}\", id(id) fcolor(Blues) clmethod(unique) legenda(off) leglabel(lev)"));
    s.run(&format!("graph export \"{svg}\", replace"));
    let text = std::fs::read_to_string(&svg).unwrap();
    assert!(!text.contains("low"), "legend should be off");
}
