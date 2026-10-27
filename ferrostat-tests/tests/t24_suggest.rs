//! "Did you mean …?" hints for misspelt commands, variables and options.

use ferrostat_tests::*;

fn err(s: &mut Session, line: &str) -> String {
    s.try_run(line).unwrap_err()
}

#[test]
fn misspelt_command() {
    let mut s = Session::with_fixture();
    let e = err(&mut s, "regression y x");
    assert!(e.contains("command regression is unrecognized\n(did you mean regress?)\nr(199);"), "{e}");
    assert!(err(&mut s, "sumarize x").contains("(did you mean summarize?)"));
    assert!(!err(&mut s, "qwertyuiop x").contains("did you mean"));
}

#[test]
fn misspelt_variable() {
    let mut s = Session::with_fixture();
    assert!(err(&mut s, "summarize gruop").contains("(did you mean group?)"));
    assert!(err(&mut s, "gen w = 2*cnnt").contains("(did you mean cnt?)"));
}

#[test]
fn misspelt_option_of_user_program() {
    let mut s = Session::with_fixture();
    s.script("program p\nsyntax [, Scale(real 1) Detail]\nend");
    assert!(err(&mut s, "p, scael(2)").contains("(did you mean scale()?)"));
    assert!(err(&mut s, "p, detial").contains("(did you mean detail?)"));
}

#[test]
fn user_programs_are_candidates_and_capture_keeps_rc() {
    let mut s = Session::new();
    s.script("program mycommand\ndi 1\nend");
    assert!(err(&mut s, "mycomand").contains("(did you mean mycommand?)"));
    s.run("capture mycomand");
    assert_eq!(s.scalar("_rc"), 199.0);
}
