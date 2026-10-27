//! System interaction: cf, describe replace/using, creturn, dir, package
//! commands, nested compound quotes in display, custom date formats.

use ferrostat_tests::*;

#[test]
fn cf_silent_on_match_and_rc9_on_mismatch() {
    let dir = temp_dir("cf");
    let f = dir.join("a.dta").display().to_string();
    let mut s = Session::with_fixture();
    s.run(&format!("save \"{f}\", replace"));
    assert_eq!(s.run(&format!("cf * using \"{f}\"")).trim(), "");
    let all = s.run(&format!("cf x name using \"{f}\", all"));
    assert!(all.contains("x:  match") && all.contains("name:  match"));
    s.run("replace x = x + 1 in 1/2");
    let e = s.try_run(&format!("cf _all using \"{f}\"")).unwrap_err();
    assert!(e.contains("x:  2 mismatches"), "{e}");
    assert!(e.contains("r(9);"));
    assert!(!e.contains("name:"));
    s.run("gen extra = 1");
    let e = s.try_run(&format!("cf extra using \"{f}\"")).unwrap_err();
    assert!(e.contains("does not exist in using"));
}

#[test]
fn describe_replace_builds_variable_dataset() {
    let mut s = Session::with_fixture();
    assert_eq!(s.run("describe, replace").trim(), "");
    assert_eq!(s.varnames(), ["position", "name", "type", "isnumeric", "format", "vallab", "varlab"]);
    assert_eq!(s.n(), 8);
    assert_eq!(s.strs("name")[1], "name");
    assert_eq!(s.strs("type")[1], "str4");
    assert_eq!(s.nums("isnumeric")[..3], [1.0, 0.0, 1.0]);
    assert_eq!(s.strs("vallab")[6], "yesno");
    assert_eq!(s.strs("varlab")[6], "Binary outcome");
    s.run("drop position format");
    assert!(!s.has_var("position"));
}

#[test]
fn describe_using_does_not_load() {
    let dir = temp_dir("descusing");
    let f = dir.join("fx").display().to_string();
    let mut s = Session::with_fixture();
    s.run(&format!("save \"{f}\", replace"));
    s.run("keep id");
    let out = s.run(&format!("describe x y using \"{f}\""));
    assert!(out.contains("Observations:") && out.contains("\ny ") && !out.contains("\nz "));
    assert_eq!(s.varnames(), ["id"]);
    s.run(&format!("describe using \"{f}\", replace"));
    assert_eq!(s.n(), 8);
}

#[test]
fn creturn_list_and_c_values() {
    let mut s = Session::with_fixture();
    let out = s.run("creturn list");
    for k in ["c(current_date)", "c(N) = 12", "c(k) = 8", "c(level) = 95", "c(type) = \"float\"", "c(os)", "c(pwd)", "c(seed)"] {
        assert!(out.contains(k), "missing {k}");
    }
    assert_eq!(s.scalar("c(width)"), 2.0 + 4.0 + 1.0 + 8.0 * 3.0 + 1.0 + 2.0);
    s.run("set linesize 120");
    assert_eq!(s.scalar("c(linesize)"), 120.0);
    assert_eq!(s.run("display c(varabbrev)").trim(), "on");
    assert_eq!(s.run("display c(dp)").trim(), "period");
    assert!(s.scalar("c(processors)") >= 1.0);
}

#[test]
fn display_nested_compound_quotes() {
    let mut s = Session::with_fixture();
    s.run("levelsof name if id <= 2, local(cn)");
    assert_eq!(s.run("display `\"`cn'\"'").trim(), "`\"ant\"' `\"bee\"'");
    assert_eq!(s.run("display `\"a `\"b\"' c\"' \"d\"").trim(), "a `\"b\"' cd");
}

#[test]
fn date_format_four_digit_year() {
    let mut s = Session::new();
    assert_eq!(s.run("display %tdYYYYNNDD 22000").trim(), "20200326");
    s.run("global date: di %tdYYYYNNDD 22000");
    assert_eq!(s.run("display \"$date\"").trim(), "20200326");
}

#[test]
fn dir_lists_files_with_sizes() {
    let dir = temp_dir("dirlist");
    let mut s = Session::with_fixture();
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    s.run(&format!("save \"{}\"", dir.join("one.dta").display()));
    let out = s.run(&format!("dir \"{}/*\"", dir.display()));
    assert!(out.contains("<dir>") && out.contains("sub/"), "{out}");
    assert!(out.lines().any(|l| l.ends_with("one.dta") && l.trim_start().contains('k')));
    let out = s.run(&format!("dir \"{}/*.dta\"", dir.display()));
    assert!(!out.contains("sub/"));
}

#[test]
fn packages_for_native_commands() {
    let mut s = Session::new();
    assert!(s.run("ssc install outreg2, replace").contains("natively"));
    s.run("ssc install ietoolkit");
    s.run("net install iefieldkit, from(\"https://example.org\")");
    s.run("ssc update");
    s.run("ado dir");
    s.run("adopath");
    s.fails("ssc install surely_not_a_package");
    assert!(s.run("which regress").contains("built-in"));
    s.fails("which surely_not_a_command");
}
