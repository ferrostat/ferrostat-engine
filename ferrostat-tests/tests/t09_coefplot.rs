//! coefplot: coefficient plots from stored estimates and matrices.

use ferrostat_tests::{temp_dir, Session};

/// Export the current graph as SVG and return its text.
fn svg(s: &mut Session, name: &str) -> String {
    let dir = temp_dir(&format!("coefplot-{name}"));
    let f = dir.join("g.svg");
    s.run(&format!("graph export \"{}\", replace", f.display()));
    std::fs::read_to_string(&f).unwrap()
}

#[test]
fn coefplot_active_estimates_uses_labels() {
    let mut s = Session::with_fixture();
    s.run("label variable x \"Slope term\"");
    s.run("regress y x z");
    s.run("coefplot");
    let text = svg(&mut s, "active");
    assert!(text.contains("Slope term"));
    assert!(text.contains("Constant"));
    let d = s.run("graph describe");
    assert!(d.contains("range") && d.contains("scatter"), "{d}");
}

#[test]
fn coefplot_keep_drop_and_vertical() {
    let mut s = Session::with_fixture();
    s.run("regress y x z");
    s.run("coefplot, keep(x z) vertical yline(0) nolabels");
    let text = svg(&mut s, "keep");
    assert!(!text.contains("Constant") && !text.contains("_cons"));
    s.run("coefplot, drop(_cons) xline(0)");
    let text = svg(&mut s, "drop");
    assert!(!text.contains("Constant"));
}

#[test]
fn coefplot_models_legend_and_subgraphs() {
    let mut s = Session::with_fixture();
    s.script("quietly regress y x\nestimates store m1\nquietly regress y x z\nestimates store m2");
    s.run("coefplot m1 m2, keep(x) vertical");
    let text = svg(&mut s, "legend");
    assert!(text.contains("m1") && text.contains("m2"));
    s.run("coefplot m1 m2, keep(x) recast(bar) citop ciopts(recast(rcap)) legend(order(1 \"First\" 3 \"Second\"))");
    let text = svg(&mut s, "order");
    assert!(text.contains("First") && text.contains("Second"));
    s.run("coefplot m1 || m2, keep(x) byopts(row(1))");
    assert!(s.run("graph describe").contains("2 combined graphs"));
    // `*` takes every stored result; plotlabels() names the plots
    s.run("coefplot *, keep(x) plotlabels(\"Short\" \"Long\")");
    let text = svg(&mut s, "star");
    assert!(text.contains("Short") && text.contains("Long"));
}

#[test]
fn coefplot_equations_groups_and_coeflabels() {
    let mut s = Session::with_fixture();
    s.script("quietly regress y x\nestimates store a1\nquietly regress z x\nestimates store a2");
    s.run("coefplot a1 a2, keep(x) aseq eqstrict eqlabels(\"Outcome Y\" \"Outcome Z\") coeflabels(x=\"Effect\") nokey");
    let text = svg(&mut s, "eq");
    assert!(text.contains("Outcome Y") && text.contains("Outcome Z") && text.contains("Effect"));
    s.run("coefplot a1 a2, groups(x=\"Main effects\") nooffset fcolor(*.5) recast(bar)");
    assert!(svg(&mut s, "groups").contains("Main effects"));
}

#[test]
fn coefplot_from_matrices() {
    let mut s = Session::new();
    s.run("matrix A = J(3, 3, .)");
    for (i, row) in [[1.0, 0.5, 1.5], [2.0, 1.2, 2.8], [0.3, -0.1, 0.7]].iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            s.run(&format!("matrix A[{},{}] = {v}", i + 1, j + 1));
        }
    }
    s.run("coefplot (matrix(A[.,1]), ci((2 3))), coeflabels(r1=\"First\" r2=\"Second\" r3=\"Third\")");
    let text = svg(&mut s, "matrix");
    assert!(text.contains("First") && text.contains("Third"));
}

#[test]
fn coefplot_errors_and_install() {
    let mut s = Session::with_fixture();
    s.fails("coefplot");
    s.run("regress y x");
    s.fails("coefplot nosuchmodel");
    s.fails("coefplot, keep(nomatch)");
    s.run("ssc install coefplot");
}

#[test]
fn coefplot_marker_labels_and_addplot() {
    let mut s = Session::with_fixture();
    s.run("regress y x z");
    s.run("coefplot, drop(_cons) mlabel format(%9.2f)");
    let b = s.scalar("_b[x]");
    assert!(svg(&mut s, "mlabel").contains(&format!("{b:.2}")));
    s.run("coefplot, drop(_cons) recast(bar) mlabcolor(none) addplot(scatter @at @b, ms(i))");
    assert!(s.run("graph describe").contains("scatter"));
}
