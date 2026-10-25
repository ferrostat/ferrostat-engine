//! Expression evaluation semantics, exercised through `generate`.

use ferrostat_tests::*;

/// Evaluate `e` for every observation of the fixture.
fn evalv(e: &str) -> Vec<f64> {
    let mut s = Session::with_fixture();
    s.run(&format!("generate double r = {e}"));
    s.nums("r")
}

/// Evaluate a constant expression (first observation).
fn val(e: &str) -> f64 {
    evalv(e)[0]
}

fn sgen(e: &str) -> Vec<String> {
    let mut s = Session::with_fixture();
    s.run(&format!("generate r = {e}"));
    s.strs("r")
}

// ---- arithmetic -----------------------------------------------------------

#[test]
fn arithmetic_on_variables() {
    let r = evalv("x * 2 + y");
    for i in 0..12 {
        assert_close(r[i], X[i] * 2.0 + Y[i], 1e-12);
    }
}

#[test]
fn arithmetic_operators() {
    assert_eq!(val("7 + 3"), 10.0);
    assert_eq!(val("7 - 3"), 4.0);
    assert_eq!(val("7 * 3"), 21.0);
    assert_close(val("7 / 2"), 3.5, 1e-15);
    assert_eq!(val("2 ^ 10"), 1024.0);
    assert_eq!(val("(1 + 2) * 3"), 9.0);
}

#[test]
fn division_by_zero_is_missing() {
    assert!(val("1 / 0").is_nan());
}

#[test]
fn missing_propagates_through_arithmetic() {
    assert!(val(". + 1").is_nan());
    assert!(val("2 * .").is_nan());
}

#[test]
fn negative_power_precedence() {
    assert_eq!(val("-2^2"), -4.0);
}

// ---- relational & logical ---------------------------------------------------

#[test]
fn relational_results_are_zero_one() {
    let r = evalv("x > 6");
    assert_eq!(r, [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.]);
    assert_eq!(val("3 >= 3"), 1.0);
    assert_eq!(val("3 <= 2"), 0.0);
    assert_eq!(val("3 == 3"), 1.0);
    assert_eq!(val("3 != 3"), 0.0);
}

#[test]
fn logical_operators() {
    assert_eq!(val("1 & 0"), 0.0);
    assert_eq!(val("1 | 0"), 1.0);
    assert_eq!(val("!0"), 1.0);
    assert_eq!(val("!5"), 0.0);
    assert_eq!(val("2 & 3"), 1.0);
}

#[test]
fn missing_is_greater_than_any_number() {
    assert_eq!(val(". > 1e300"), 1.0);
    assert_eq!(val(". == ."), 1.0);
    assert_eq!(val(". < 5"), 0.0);
}

#[test]
fn missing_is_true_in_logical_context() {
    assert_eq!(val(". & 1"), 1.0);
    assert_eq!(val("!."), 0.0);
}

#[test]
fn string_comparison() {
    let r = evalv("name == \"cat\"");
    assert_eq!(r.iter().filter(|&&v| v == 1.0).count(), 1);
    assert_eq!(r[2], 1.0);
    assert_eq!(evalv("name < \"c\"")[..3], [1., 1., 0.]);
}

#[test]
fn string_number_comparison_is_type_mismatch() {
    let mut s = Session::with_fixture();
    s.fails("generate r = name == 1");
}

// ---- math functions -------------------------------------------------------

#[test]
fn math_functions() {
    assert_close(val("abs(-3.5)"), 3.5, 0.0);
    assert_close(val("exp(1)"), std::f64::consts::E, 1e-15);
    assert_close(val("ln(100)"), 100f64.ln(), 1e-15);
    assert_close(val("log(100)"), 100f64.ln(), 1e-15);
    assert_close(val("log10(1000)"), 3.0, 1e-15);
    assert_close(val("sqrt(16)"), 4.0, 0.0);
    assert_close(val("sin(0)"), 0.0, 0.0);
    assert_close(val("cos(0)"), 1.0, 0.0);
    assert_close(val("tan(0)"), 0.0, 0.0);
    assert_eq!(val("floor(-1.5)"), -2.0);
    assert_eq!(val("ceil(-1.5)"), -1.0);
    assert_eq!(val("int(-1.7)"), -1.0);
    assert_eq!(val("sign(-4)"), -1.0);
}

#[test]
fn domain_errors_give_missing() {
    assert!(val("ln(0)").is_nan());
    assert!(val("ln(-1)").is_nan());
    assert!(val("sqrt(-1)").is_nan());
}

#[test]
fn round_one_argument() {
    assert_eq!(val("round(2.5)"), 3.0);
    assert_eq!(val("round(-2.5)"), -3.0);
    assert_eq!(val("round(2.4)"), 2.0);
}

#[test]
fn round_two_arguments() {
    assert_close(val("round(3.14159, 0.01)"), 3.14, 1e-12);
    assert_eq!(val("round(17, 5)"), 15.0);
}

#[test]
fn row_min_max_over_arguments() {
    assert_eq!(val("max(3, 9, 1)"), 9.0);
    assert_eq!(val("min(3, 9, 1)"), 1.0);
}

#[test]
fn more_math_functions() {
    assert_eq!(val("mod(17, 5)"), 2.0);
    assert_eq!(val("comb(5, 2)"), 10.0);
    assert_close(val("lnfactorial(5)"), 120f64.ln(), 1e-12);
    assert_close(val("invlogit(0)"), 0.5, 1e-15);
    assert_close(val("logit(0.5)"), 0.0, 1e-15);
    assert_close(val("atan(1)"), std::f64::consts::FRAC_PI_4, 1e-15);
}

#[test]
fn distribution_functions() {
    assert_close(val("normal(1.96)"), 0.9750021048517795, 1e-9);
    assert_close(val("invnormal(0.975)"), 1.959963984540054, 1e-9);
    assert_close(val("normalden(0)"), 0.3989422804014327, 1e-12);
    assert_close(val("ttail(10, 2.228)"), 0.025, 1e-3);
    assert_close(val("chi2tail(1, 3.841)"), 0.05, 1e-3);
    assert_close(val("Ftail(2, 9, 4.256)"), 0.05, 1e-3);
    assert_close(val("binomial(10, 3, 0.5)"), 176.0 / 1024.0, 1e-12);
}

#[test]
fn programming_functions() {
    let mut s = Session::with_missing_fixture();
    s.run("generate m = missing(y)");
    assert_eq!(s.nums("m")[11], 1.0);
    assert_eq!(s.nums("m")[0], 0.0);
    assert_eq!(val("inlist(3, 1, 2, 3)"), 1.0);
    assert_eq!(val("inrange(5, 1, 4)"), 0.0);
    assert_eq!(val("cond(1 > 0, 10, 20)"), 10.0);
    assert_eq!(evalv("cond(x > 6, 1, 0)")[11], 1.0);
}

#[test]
fn random_numbers_are_reproducible() {
    let mut s = Session::with_fixture();
    s.run("set seed 12345");
    s.run("generate u1 = runiform()");
    s.run("set seed 12345");
    s.run("generate u2 = runiform()");
    assert_eq!(s.nums("u1"), s.nums("u2"));
    assert!(s.nums("u1").iter().all(|&u| (0.0..1.0).contains(&u)));
    s.run("generate n1 = rnormal()");
}

// ---- system variables -------------------------------------------------------

#[test]
fn underscore_n_and_big_n() {
    assert_eq!(evalv("_n"), X);
    assert_eq!(evalv("_N"), [12.0; 12]);
}

#[test]
fn explicit_subscripts() {
    let r = evalv("y[_n-1]");
    assert!(r[0].is_nan());
    assert_eq!(r[1], Y[0]);
}

#[test]
fn builtin_constants() {
    assert_close(val("_pi"), std::f64::consts::PI, 1e-15);
    assert_eq!(val("c(N)"), 12.0);
}

// ---- string functions -------------------------------------------------------

#[test]
fn string_literal_assignment() {
    assert_eq!(sgen("\"hello\"")[0], "hello");
}

#[test]
fn string_concatenation() {
    assert_eq!(sgen("name + \"-\" + \"x\"")[0], "ant-x");
}

#[test]
fn string_functions() {
    assert_eq!(evalv("strlen(name)")[8], 4.0);
    assert_eq!(sgen("substr(name, 2, 2)")[0], "nt");
    assert_eq!(sgen("upper(name)")[0], "ANT");
    assert_eq!(sgen("lower(\"ABC\")")[0], "abc");
    assert_eq!(sgen("strtrim(\"  a  \")")[0], "a");
    assert_eq!(evalv("strpos(name, \"n\")")[0], 2.0);
    assert_eq!(sgen("subinstr(name, \"a\", \"A\", .)")[0], "Ant");
    assert_eq!(sgen("word(\"a b c\", 2)")[0], "b");
    assert_eq!(val("real(\"3.5\")"), 3.5);
    assert_eq!(sgen("string(42)")[0], "42");
}

#[test]
fn regex_functions() {
    assert_eq!(evalv("regexm(name, \"^[a-c]\")")[..4], [1., 1., 1., 0.]);
    assert_eq!(sgen("ustrregexra(name, \"[aeiou]\", \"\")")[0], "nt");
}

// ---- dates ------------------------------------------------------------------

#[test]
fn date_functions() {
    // dates count days since 01jan1960.
    assert_eq!(val("mdy(1, 1, 1960)"), 0.0);
    assert_eq!(val("mdy(1, 1, 2000)"), 14610.0);
    assert_eq!(val("date(\"2000-01-01\", \"YMD\")"), 14610.0);
    assert_eq!(val("year(14610)"), 2000.0);
    assert_eq!(val("month(14640)"), 1.0);
    assert_eq!(val("dow(0)"), 5.0); // 1 Jan 1960 was a Friday
}
