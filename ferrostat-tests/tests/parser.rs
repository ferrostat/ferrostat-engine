//! Command-line syntax: verb, varlist, if/in, weights, using, options,
//! expression grammar and lexical details.

use ferrostat_parser::parser::{parse_line, Parser};
use ferrostat_parser::{BinOp, Expr, InRange, UnaryOp};

fn expr(s: &str) -> Expr {
    Parser::new(s).parse_expr().unwrap_or_else(|e| panic!("cannot parse `{s}`: {e}"))
}

fn num(n: f64) -> Box<Expr> {
    Box::new(Expr::Number(n))
}

#[test]
fn verb_and_varlist() {
    let c = parse_line("summarize price mpg weight").unwrap();
    assert_eq!(c.verb, "summarize");
    assert_eq!(c.args, ["price", "mpg", "weight"]);
}

#[test]
fn verb_is_case_insensitive() {
    assert_eq!(parse_line("SUMMARIZE x").unwrap().verb, "summarize");
}

#[test]
fn empty_line_is_empty_command() {
    assert_eq!(parse_line("").unwrap().verb, "");
    assert_eq!(parse_line("   ").unwrap().verb, "");
}

#[test]
fn if_qualifier() {
    let c = parse_line("summarize x if y > 3 & z != 2").unwrap();
    assert_eq!(c.args, ["x"]);
    assert!(matches!(c.if_expr, Some(Expr::Binary(_, BinOp::And, _))));
}

#[test]
fn in_qualifier_single_and_range() {
    assert_eq!(parse_line("list in 5").unwrap().in_range, Some(InRange::Single(5)));
    assert_eq!(parse_line("list x in 2/7").unwrap().in_range, Some(InRange::Range(2, 7)));
}

#[test]
#[ignore = "missing: `in f/l`, `in -5/l` and other first/last forms"]
fn in_qualifier_first_last() {
    assert_eq!(parse_line("list in f/5").unwrap().in_range, Some(InRange::First(5)));
    assert_eq!(parse_line("list in -3/l").unwrap().in_range, Some(InRange::Last(3)));
}

#[test]
fn if_and_in_together() {
    let c = parse_line("list x in 1/10 if y > 2").unwrap();
    assert_eq!(c.in_range, Some(InRange::Range(1, 10)));
    assert!(c.if_expr.is_some());
}

#[test]
fn weights() {
    for wt in ["fw", "aw", "pw", "iw", "fweight", "aweight", "pweight", "iweight"] {
        let c = parse_line(&format!("regress y x [{wt}=z]")).unwrap();
        let w = c.weight.unwrap_or_else(|| panic!("weight {wt} not parsed"));
        assert_eq!(w.weight_type, wt);
        assert_eq!(w.expr, Expr::Var("z".into()));
        assert_eq!(c.args, ["y", "x"]);
    }
}

#[test]
fn using_clause() {
    let c = parse_line("merge 1:1 id using /tmp/other.dta, keep(match)").unwrap();
    assert_eq!(c.using.as_deref(), Some("/tmp/other.dta"));
    assert_eq!(c.options.len(), 1);
}

#[test]
fn using_quoted_path() {
    let c = parse_line("append using \"my file.dta\"").unwrap();
    assert_eq!(c.using.as_deref(), Some("my file.dta"));
}

#[test]
fn options_plain_with_args_and_multiple() {
    let c = parse_line("regress y x, robust level(90) noconstant").unwrap();
    let names: Vec<&str> = c.options.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["robust", "level", "noconstant"]);
    assert_eq!(c.options[1].1, Some(Expr::Number(90.0)));
}

#[test]
#[ignore = "missing: option arguments that are not a single expression, e.g. by(a b) or title(\"x\") with spaces"]
fn option_with_varlist_argument() {
    let c = parse_line("collapse (mean) y, by(group d)").unwrap();
    let by = c.options.iter().find(|(n, _)| n == "by").expect("by option");
    assert!(by.1.is_some());
}

#[test]
fn options_without_commas_between() {
    let c = parse_line("tabulate a b, chi2 row col").unwrap();
    assert_eq!(c.options.len(), 3);
}

#[test]
fn arithmetic_precedence() {
    // 1 + 2 * 3 parses as 1 + (2 * 3)
    assert_eq!(
        expr("1 + 2 * 3"),
        Expr::Binary(num(1.0), BinOp::Add, Box::new(Expr::Binary(num(2.0), BinOp::Mul, num(3.0))))
    );
}

#[test]
fn power_is_right_associative() {
    assert_eq!(
        expr("2^3^2"),
        Expr::Binary(num(2.0), BinOp::Pow, Box::new(Expr::Binary(num(3.0), BinOp::Pow, num(2.0))))
    );
}

#[test]
fn subtraction_is_left_associative() {
    assert_eq!(
        expr("8 - 4 - 2"),
        Expr::Binary(Box::new(Expr::Binary(num(8.0), BinOp::Sub, num(4.0))), BinOp::Sub, num(2.0))
    );
}

#[test]
#[ignore = "bug: -2^2 parses as (-2)^2 = 4; Stata gives -4"]
fn unary_minus_binds_looser_than_power() {
    // In Stata -2^2 == -4
    assert_eq!(
        expr("-2^2"),
        Expr::Unary(UnaryOp::Neg, Box::new(Expr::Binary(num(2.0), BinOp::Pow, num(2.0))))
    );
}

#[test]
fn logical_precedence_and_before_or() {
    match expr("a | b & c") {
        Expr::Binary(_, BinOp::Or, rhs) => assert!(matches!(*rhs, Expr::Binary(_, BinOp::And, _))),
        e => panic!("unexpected {e:?}"),
    }
}

#[test]
fn relational_operators() {
    for (src, op) in [("a < b", BinOp::Lt), ("a <= b", BinOp::Le), ("a > b", BinOp::Gt),
                      ("a >= b", BinOp::Ge), ("a == b", BinOp::Eq), ("a != b", BinOp::Ne), ("a ~= b", BinOp::Ne)] {
        assert!(matches!(expr(src), Expr::Binary(_, ref o, _) if *o == op), "{src}");
    }
}

#[test]
fn negation_operators() {
    assert!(matches!(expr("!a"), Expr::Unary(UnaryOp::Not, _)));
    assert!(matches!(expr("~a"), Expr::Unary(UnaryOp::Not, _)));
}

#[test]
fn function_calls() {
    assert_eq!(expr("max(a, 2, b)"), Expr::Call("max".into(), vec![Expr::Var("a".into()), Expr::Number(2.0), Expr::Var("b".into())]));
    assert_eq!(expr("runiform()"), Expr::Call("runiform".into(), vec![]));
}

#[test]
fn literals() {
    assert_eq!(expr("."), Expr::Missing);
    assert_eq!(expr("\"hello\""), Expr::String("hello".into()));
    assert_eq!(expr("1.5e3"), Expr::Number(1500.0));
    assert_eq!(expr(".5"), Expr::Number(0.5));
}

#[test]
#[ignore = "missing: extended missing literals .a-.z in expressions"]
fn extended_missing_literal() {
    assert!(!matches!(expr(".a"), Expr::Number(_)));
    assert_ne!(expr(".a"), expr(".b"));
}

#[test]
#[ignore = "missing: compound double quotes `\"...\"'"]
fn compound_quotes() {
    assert_eq!(expr("`\"say \"hi\"\"'"), Expr::String("say \"hi\"".into()));
}

#[test]
#[ignore = "missing: explicit subscripts x[_n-1]"]
fn subscripts() {
    assert!(matches!(expr("x[_n-1]"), Expr::Index(_, _)));
}

#[test]
fn unbalanced_parens_is_error() {
    assert!(Parser::new("(1 + 2").parse_expr().is_err());
}

#[test]
fn line_comments_are_ignored() {
    let c = parse_line("summarize x // trailing comment").unwrap();
    assert_eq!(c.args, ["x"]);
}

#[test]
fn block_comments_are_ignored() {
    let c = parse_line("summarize /* note */ x").unwrap();
    assert_eq!(c.args, ["x"]);
}

#[test]
fn global_macro_token() {
    let c = parse_line("summarize $controls").unwrap();
    assert!(!c.raw_args.is_empty());
}

#[test]
fn raw_line_is_preserved() {
    let line = "generate w = x * 2 if y > 1";
    assert_eq!(parse_line(line).unwrap().raw, line);
}

#[test]
#[ignore = "missing: varlist abbreviations and wildcards (x*, x?, a-c) are expanded by the parser/engine"]
fn varlist_wildcards() {
    let c = parse_line("summarize x*").unwrap();
    assert_eq!(c.args, ["x*"]);
}
