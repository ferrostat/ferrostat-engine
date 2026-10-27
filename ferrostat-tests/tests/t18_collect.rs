//! The table system: `table` statistics landing in collections, `collect`
//! (prefix, get, labels, styles, layouts, export), `dtable` and `etable`.

use ferrostat_tests::*;

/// The data row of a table whose header text starts with `label`.
fn line_of<'a>(out: &'a str, label: &str) -> &'a str {
    out.lines().find(|l| l.trim_start().starts_with(label)).unwrap_or_else(|| panic!("no row {label} in\n{out}"))
}

#[test]
fn table_percent_across_gives_row_percentages() {
    let mut s = Session::with_fixture();
    let out = s.run("table group d, statistic(percent, across(d))");
    // group 1 has d = 0,0,1,0: 75% / 25%
    let r1 = numbers(line_of(&out, "1 "));
    assert_eq!(r1, vec![1.0, 75.0, 25.0, 100.0]);
    let tot = numbers(line_of(&out, "Total"));
    assert_close(tot[0], 100.0 * 6.0 / 12.0, 1e-9);
    assert_eq!(*tot.last().unwrap(), 100.0);
    // default formats: percent with two decimals, frequency with commas
    assert!(out.contains("75.00"));
    let out = s.run("table group, statistic(frequency) statistic(percent)");
    assert!(out.contains("Frequency") && out.contains("Percent") && out.contains("33.33"));
}

#[test]
fn table_factor_variable_statistics_and_formats() {
    let mut s = Session::with_fixture();
    let out = s.run("table (var) (group), stat(count x) stat(mean x) stat(sd x) stat(fvpercent d) nformat(%5.2f mean sd) nformat(%5.1f fvpercent) sformat((%s) sd) sformat(%s%% fvpercent) style(table-1)");
    // fvpercent of d within group 2 (0,1,0,1): 50%
    assert!(out.contains("Binary outcome"));
    let yes = line_of(&out, "yes");
    assert!(yes.contains("25.0%") && yes.contains("50.0%") && yes.contains("75.0%"), "{yes}");
    // sd in parentheses under the mean
    s.run("quietly summarize x if group == 1");
    let sd = s.scalar("r(sd)");
    assert!(out.contains(&format!("({sd:.2})")), "{out}");
    let out = s.run("table group, stat(fvfrequency d)");
    assert!(out.contains("Binary outcome"));
}

#[test]
fn table_results_land_in_the_collection() {
    let mut s = Session::with_fixture();
    s.run("quietly table group, statistic(frequency) statistic(percent)");
    s.run(r#"collect label levels result frequency "N" percent "Pct", modify"#);
    s.run(r#"collect title "Groups""#);
    let out = s.run("collect preview");
    assert!(out.starts_with("Groups"));
    assert!(out.contains(" N ") || out.contains("N  ") && out.contains("Pct"));
    s.run("collect style header result, level(hide)");
    s.run("collect style cell border_block, border(right, pattern(nil))");
    let out = s.run("collect preview");
    assert!(!out.contains('|') && !out.contains("Pct"));
    let out = s.run("collect layout (group) (result[percent])");
    assert!(out.contains("Collection: Table") && out.contains("Table 1: 4 x 1"));
}

#[test]
fn collect_prefix_with_tags_and_autolevels() {
    let mut s = Session::with_fixture();
    s.run("collect clear");
    s.run("quietly collect, tags(cmdset[x]): summarize x, detail");
    s.run("quietly collect, tags(cmdset[y]): summarize y, detail");
    s.run("collect style autolevels result min p50 max");
    s.run(r#"collect label levels result min "Min" p50 "Md" max "Max", modify"#);
    let out = s.run("collect layout (cmdset) (result)");
    assert!(out.contains("Table 1: 2 x 3"));
    assert!(out.lines().any(|l| l.contains("Min") && l.contains("Md") && l.contains("Max")));
    s.run("quietly summarize y, detail");
    let want = vec![s.scalar("r(min)"), s.scalar("r(p50)"), s.scalar("r(max)")];
    assert_all_close(&numbers(line_of(&out, "y ")), &want, 1e-9);
    let out = s.run("collect levelsof cmdset");
    assert!(out.contains("Levels: x y"));
    assert!(s.run("collect dims").contains("cmdset"));
}

#[test]
fn collect_regression_results() {
    let mut s = Session::with_fixture();
    s.run("collect clear");
    s.run("quietly collect: regress y x");
    s.run("quietly collect: regress y x z");
    s.run("collect style cell result[_r_b _r_se], nformat(%9.4f)");
    let out = s.run("collect layout (colname#result[_r_b _r_se]) (cmdset)");
    let b = s.scalar("_b[z]");
    assert!(out.contains(&format!("{b:.4}")), "{out}");
    assert!(out.contains("Intercept"));
    // collect get after the fact
    s.run("collect create other");
    s.run("quietly summarize x");
    s.run("collect get r(mean sd), tags(var[x])");
    let out = s.run("collect layout (var) (result)");
    assert_all_close(&numbers(line_of(&out, "x ")), &[6.5, s.scalar("r(sd)")], 1e-6);
}

#[test]
fn dtable_by_with_tests() {
    let mut s = Session::with_fixture();
    let out = s.run("dtable y x i.group, by(d, tests)");
    assert!(out.contains("note: using test regress across levels of d for y and x."));
    assert!(out.contains("note: using test pearson across levels of d for group."));
    assert!(out.contains("Binary outcome") && out.contains("Test") && out.contains("Total"));
    // sample row: frequency (percent)
    assert!(line_of(&out, "N ").contains("6 (50.0%)"));
    // mean (sd) of y in d == 0
    s.run("quietly summarize y if d == 0");
    let (m, sd) = (s.scalar("r(mean)"), s.scalar("r(sd)"));
    assert!(line_of(&out, "y ").contains(&format!("{m:.3} ({sd:.3})")), "{out}");
    // regress test equals the one-way ANOVA F test
    s.run("quietly regress y i.d");
    let p = s.scalar("Ftail(e(df_m), e(df_r), e(F))");
    assert!(line_of(&out, "y ").ends_with(&format!("{p:.3}")), "{out}");
    // Pearson chi-squared test of the factor variable
    s.run("quietly tabulate group d, chi2");
    let p = s.scalar("r(p)");
    assert!(line_of(&out, "1 ").ends_with(&format!("{p:.3}")), "{out}");
}

#[test]
fn dtable_without_by() {
    let mut s = Session::with_fixture();
    let out = s.run("dtable x, continuous(z, statistics(median iqi)) title(Descriptives)");
    assert!(out.starts_with("Descriptives"));
    assert!(out.contains("Summary"));
    assert!(numbers(line_of(&out, "N ")) == vec![12.0]);
    s.run("quietly summarize z, detail");
    let med = s.scalar("r(p50)");
    assert!(line_of(&out, "z ").contains(&format!("{med:.3} [")), "{out}");
}

#[test]
fn etable_side_by_side() {
    let mut s = Session::with_fixture();
    s.run("quietly regress y x");
    s.run("estimates store m1");
    s.run("quietly regress y x i.d");
    s.run("estimates store m2");
    let out = s.run("etable, estimates(m1 m2) column(estimates) mstat(N) mstat(r2) showstars showstarsnote");
    assert!(out.contains("m1") && out.contains("m2"));
    let b = s.scalar("_b[x]");
    let se = s.scalar("_se[x]");
    let x = line_of(&out, "x ");
    assert!(x.contains(&format!("{b:.3} **")), "{out}");
    assert!(out.contains(&format!("({se:.3})")));
    assert!(out.contains("Binary outcome") && line_of(&out, "yes").len() > 5);
    assert!(numbers(line_of(&out, "Number of observations")) == vec![12.0, 12.0]);
    assert!(out.contains("R-squared"));
    assert!(out.trim_end().ends_with("** p<.01, * p<.05"));
    // z statistics with a custom format, the active estimates
    let out = s.run("etable, cstat(_r_b) cstat(_r_z, sformat((%s))) note(t statistics in parentheses)");
    let t = s.scalar("_b[x]/_se[x]");
    assert!(out.contains(&format!("({t:.2})")), "{out}");
    assert!(out.contains("t statistics in parentheses"));
    let out = s.run("etable, keep(x) cstat(_r_b)");
    assert!(!out.contains("Intercept"));
}

#[test]
fn collect_export_formats() {
    let mut s = Session::with_fixture();
    let dir = temp_dir("t18_export");
    s.run("quietly table group d");
    for ext in ["html", "md", "txt", "xlsx", "docx", "tex"] {
        let f = dir.join(format!("tab.{ext}"));
        let out = s.run(&format!("collect export \"{}\", replace", f.display()));
        assert!(out.contains("exported"), "{out}");
        assert!(std::fs::metadata(&f).unwrap().len() > 50, "{ext}");
    }
    let html = std::fs::read_to_string(dir.join("tab.html")).unwrap();
    assert!(html.contains("<table>") && html.contains("Total"));
    let md = std::fs::read_to_string(dir.join("tab.md")).unwrap();
    assert!(md.contains("| Total |"));
    let f = dir.join("d.docx");
    s.run(&format!("quietly dtable x, export(\"{}\", replace)", f.display()));
    assert!(f.exists());
    s.fails(&format!("collect export \"{}\"", dir.join("tab.html").display()));
}

#[test]
fn collect_management() {
    let mut s = Session::with_fixture();
    s.run("collect create mine");
    s.run("quietly collect: summarize x");
    s.run("collect notes \"first\"");
    let out = s.run("collect layout (cmdset) (result[N mean])");
    assert!(out.contains("Collection: mine") && out.contains("first"));
    assert!(s.run("collect dir").contains("mine"));
    s.run("collect recode cmdset 1 = model1");
    assert!(s.run("collect preview").contains("model1"));
    s.run("collect clear");
    s.fails("collect preview");
    s.fails("collect frobnicate");
}
