//! margins with at() numlists and statistics, margins over factor levels,
//! predict() options for nonlinear models, and marginsplot.

use ferrostat_tests::*;

const TOL: f64 = 1e-6;

/// Margin `j` and its standard error from the last margins results.
fn margin(s: &mut Session, j: usize) -> (f64, f64) {
    s.run("matrix __T = r(table)");
    (s.scalar(&format!("el(__T,1,{j})")), s.scalar(&format!("el(__T,2,{j})")))
}

/// lincom of an expression: (estimate, se).
fn lincom(s: &mut Session, expr: &str) -> (f64, f64) {
    s.run(&format!("lincom {expr}"));
    (s.scalar("r(estimate)"), s.scalar("r(se)"))
}

/// A larger deterministic dataset: x continuous, g in 1..3, binary b,
/// ordered o in 1..3, count c.
fn sim(s: &mut Session) {
    s.script(
        "clear
set obs 300
gen x = mod(_n*37, 101)/10
gen g = 1 + mod(_n, 3)
gen u = mod(_n*7919, 1000)/1000
gen s = 0.4*x - 0.8*(g==2) + 0.5*(g==3) + 4*u
gen b = s > 3
gen o = 1 + (s > 2.5) + (s > 4.5)
gen c = floor(exp(0.15*x + 0.2*(g==3) + u - 0.5))",
    );
}

#[test]
fn at_numlist_after_regress_matches_lincom() {
    let mut s = Session::with_fixture();
    s.run("regress y x z");
    // previously "invalid at()"
    let out = s.run("margins, at(x=(2(2)6))");
    assert!(out.contains("Predictive margins"), "{out}");
    assert!(out.contains("1._at: x = 2") && out.contains("3._at: x = 6"), "{out}");
    assert!(out.contains("         _at |"), "{out}");
    assert!(out.contains("P>|t|"), "{out}");
    assert_eq!(s.scalar("colsof(r(b))"), 3.0);
    assert_eq!(s.scalar("r(k_at)"), 3.0);
    s.run("matrix M = r(b)");
    s.run("matrix T = r(table)");
    s.run("summarize z");
    let zbar = s.scalar("r(mean)");
    s.run("regress y x z");
    for (j, v) in [2.0, 4.0, 6.0].iter().enumerate() {
        let (e, se) = lincom(&mut s, &format!("_cons + {v}*x + {zbar}*z"));
        assert_close(s.scalar(&format!("el(M,1,{})", j + 1)), e, TOL);
        assert_close(s.scalar(&format!("el(T,2,{})", j + 1)), se, 1e-5);
    }
}

#[test]
fn at_cartesian_product_and_order() {
    let mut s = Session::with_fixture();
    s.run("regress y x z");
    let out = s.run("margins, at(x=(1 2 3) z=(0 1)) vsquish");
    assert_eq!(s.scalar("colsof(r(b))"), 6.0);
    // the last variable varies fastest
    assert!(out.contains("1._at: x = 1\n       z = 0\n2._at: x = 1\n       z = 1"), "{out}");
    assert!(out.contains("Adjusted predictions"), "{out}");
    s.run("matrix M = r(b)");
    s.run("matrix A = r(at)");
    assert_eq!(s.scalar("rowsof(A)"), 6.0);
    let (e, _) = lincom(&mut s, "_cons + 3*x + 1*z");
    assert_close(s.scalar("el(M,1,6)"), e, TOL);
    let (e, _) = lincom(&mut s, "_cons + 2*x");
    assert_close(s.scalar("el(M,1,3)"), e, TOL);
    // without vsquish multi-line blocks are separated by blank lines
    let out = s.run("margins, at(x=(1 2) z=0)");
    assert!(out.contains("       z = 0\n\n2._at:"), "{out}");
}

#[test]
fn at_statistics_and_multiple_at_options() {
    let mut s = Session::with_fixture();
    s.run("regress y x z i.group");
    let out = s.run("margins, at((mean) _all x=5) at(x=7)");
    assert!(out.contains("(mean)"), "{out}");
    assert!(out.contains("1.group") && out.contains("2._at: x"), "{out}");
    assert_eq!(s.scalar("r(k_at)"), 2.0);
    s.run("matrix M = r(b)");
    // at means of everything but x: z mean, group indicators at their shares (1/3 each)
    s.run("summarize z");
    let zbar = s.scalar("r(mean)");
    s.run("regress y x z i.group");
    let (e, _) = lincom(&mut s, &format!("_cons + 5*x + {zbar}*z + (1/3)*2.group + (1/3)*3.group"));
    assert_close(s.scalar("el(M,1,1)"), e, TOL);
    // second at(): group as observed gives the same linear margin
    assert_close(s.scalar("el(M,1,2)") - s.scalar("el(M,1,1)"), 2.0 * s.scalar("_b[x]"), TOL);
    // (median) and (p25) statistics
    let out = s.run("margins, at((median) z (p25) x)");
    assert!(out.contains("(median)") && out.contains("(p25)"), "{out}");
    // a variable outside the model is rejected
    s.fails("margins, at(cnt=1)");
}

#[test]
fn logit_atmeans_and_factor_margins() {
    let mut s = Session::new();
    sim(&mut s);
    s.run("logit b x i.g");
    let out = s.run("margins g, atmeans");
    assert!(out.contains("Adjusted predictions"), "{out}");
    assert!(out.contains("At: x   =") && out.contains("1.g = ") && out.contains("(mean)"), "{out}");
    assert!(out.contains("           g |\n          1  |"), "{out}");
    s.run("matrix M = r(b)");
    s.run("matrix T = r(table)");
    s.run("summarize x");
    let xbar = s.scalar("r(mean)");
    s.run("logit b x i.g");
    for (j, lv) in [(1usize, ""), (2, " + 2.g"), (3, " + 3.g")] {
        let (xb, sexb) = lincom(&mut s, &format!("_cons + {xbar}*x{lv}"));
        let p = 1.0 / (1.0 + (-xb).exp());
        assert_close(s.scalar(&format!("el(M,1,{j})")), p, TOL);
        assert_close(s.scalar(&format!("el(T,2,{j})")), p * (1.0 - p) * sexb, 1e-5);
    }
    // factor margins crossed with an at() numlist: _at#g rows
    let out = s.run("margins g, at(x=(2(4)10)) vsquish");
    assert!(out.contains("       _at#g |") && out.contains("        3 3  |"), "{out}");
    assert_eq!(s.scalar("colsof(r(b))"), 9.0);
    // predict(xb) gives linear predictions
    s.run("margins, at(x=4) predict(xb) atmeans");
    let (m, _) = margin(&mut s, 1);
    let (xb, _) = lincom(&mut s, "_cons + 4*x + (1/3)*2.g + (1/3)*3.g");
    assert_close(m, xb, TOL);
    // average marginal effects of a factor: discrete change
    let out = s.run("margins, dydx(g)");
    assert!(out.contains("Note: dy/dx for factor levels"), "{out}");
}

#[test]
fn probit_poisson_nbreg_predictions() {
    let mut s = Session::new();
    sim(&mut s);
    s.run("probit b x");
    s.run("margins, at(x=3) predict(pr)");
    let (m, se) = margin(&mut s, 1);
    let (xb, sexb) = lincom(&mut s, "_cons + 3*x");
    assert_close(m, s.scalar(&format!("normal({xb})")), TOL);
    assert_close(se, s.scalar(&format!("normalden({xb})")) * sexb, 1e-5);

    s.run("poisson c x i.g");
    let out = s.run("margins, at(x=(0 5)) atmeans predict(n)");
    assert!(out.contains("Predicted number of events, predict(n)"), "{out}");
    let (m, se) = margin(&mut s, 2);
    let (xb, sexb) = lincom(&mut s, "_cons + 5*x + (1/3)*2.g + (1/3)*3.g");
    assert_close(m, xb.exp(), TOL);
    assert_close(se, xb.exp() * sexb, 1e-5);

    s.run("nbreg c x");
    s.run("margins, at(x=2)");
    let (m, _) = margin(&mut s, 1);
    let (xb, _) = lincom(&mut s, "_cons + 2*x");
    assert_close(m, xb.exp(), TOL);
}

#[test]
fn ordered_and_multinomial_outcomes() {
    let mut s = Session::new();
    sim(&mut s);
    s.run("ologit o x");
    let mut total = 0.0;
    for k in 1..=3 {
        let out = s.run(&format!("margins, at(x=4) predict(outcome({k}))"));
        assert!(out.contains(&format!("Pr(o=={k}), predict(outcome({k}))")), "{out}");
        total += margin(&mut s, 1).0;
    }
    assert_close(total, 1.0, 1e-10);
    s.run("margins, at(x=4) predict(outcome(1))");
    let (m, _) = margin(&mut s, 1);
    s.run("matrix B = e(b)");
    let p = s.scalar("invlogit(el(B,1,2) - 4*el(B,1,1))");
    assert_close(m, p, TOL);
    // no outcome(): one margin per outcome
    let out = s.run("margins, at(x=4)");
    assert!(out.contains("1._predict:") && out.contains("    _predict |"), "{out}");
    assert_eq!(s.scalar("colsof(r(b))"), 3.0);

    s.run("mlogit o x i.g, base(1)");
    let mut total = 0.0;
    for k in 1..=3 {
        s.run(&format!("margins g, atmeans predict(outcome({k}))"));
        total += margin(&mut s, 2).0;
    }
    assert_close(total, 1.0, 1e-10);
}

#[test]
fn margins_post_and_r_results() {
    let mut s = Session::with_fixture();
    s.run("regress y x z");
    s.run("margins, at(x=(1 2))");
    assert_eq!(s.run("display r(cmd)").trim(), "margins");
    s.run("margins, at(x=(1 2)) post");
    assert_eq!(s.run("display e(cmd)").trim(), "margins");
    assert_eq!(s.scalar("colsof(e(b))"), 2.0);
}

/// Export the current graph as SVG and return its text.
fn svg(s: &mut Session, name: &str) -> String {
    let dir = temp_dir(&format!("marginsplot-{name}"));
    let f = dir.join("g.svg");
    s.run(&format!("graph export \"{}\", replace", f.display()));
    std::fs::read_to_string(&f).unwrap()
}

#[test]
fn marginsplot_after_at_and_factor() {
    let mut s = Session::new();
    sim(&mut s);
    s.run("logit b x i.g");
    s.fails("marginsplot");
    s.run("margins, at(x=(2(2)8))");
    s.run("marginsplot");
    let d = s.run("graph describe");
    assert!(d.contains("connected") && d.contains("range(cap)"), "{d}");
    let text = svg(&mut s, "at");
    assert!(text.contains("Predictive margins with 95% CIs"), "missing title");
    assert!(text.contains("Pr(b)"));

    // at() and factor levels: one line per level of g
    s.run("margins g, at(x=(2(2)8))");
    s.run("marginsplot, noci name(byg) title(\"By group\")");
    let d = s.run("graph describe");
    assert!(!d.contains("range"), "{d}");
    assert_eq!(d.matches("connected").count(), 3, "{d}");
    let text = svg(&mut s, "byg");
    assert!(text.contains("By group") && text.contains("g=1") && text.contains("g=3"));

    // margins over a factor: categories on the x axis; recast(bar)
    s.run("margins g");
    s.run("marginsplot, recast(bar) ytitle(\"Probability\")");
    let d = s.run("graph describe");
    assert!(d.contains("range(bar)") && d.contains("range(cap)"), "{d}");
    let text = svg(&mut s, "bar");
    assert!(text.contains("Predictive margins of g with 95% CIs") && text.contains("Probability"));

    // x() picks the dimension on the x axis
    s.run("margins g, at(x=(2 8))");
    s.run("marginsplot, x(g) recast(line) recastci(rarea)");
    let d = s.run("graph describe");
    assert!(d.contains("line") && d.contains("range(area)"), "{d}");
}
