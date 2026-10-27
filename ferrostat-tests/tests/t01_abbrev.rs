//! Command, subcommand, option and macro-function abbreviations, plus
//! `clonevar` and the interactive viewers `browse`/`edit`.

use ferrostat_tests::*;

#[test]
fn label_command_and_subcommand_abbreviations() {
    let mut s = Session::with_fixture();
    s.run("la var x \"The x\"");
    s.run("lab variable y \"The y\"");
    s.run("labe var z \"The z\"");
    s.run("la def grp 1 \"one\" 2 \"two\"");
    s.run("la de grp2 1 \"uno\"");
    s.run("la val group grp");
    s.run("label values cnt grp2");
    s.run("la da \"New data label\"");
    let ds = s.ds();
    assert_eq!(ds.get_var("x").unwrap().label, "The x");
    assert_eq!(ds.get_var("y").unwrap().label, "The y");
    assert_eq!(ds.get_var("z").unwrap().label, "The z");
    assert_eq!(ds.get_var("group").unwrap().value_label_name, "grp");
    assert_eq!(ds.get_var("cnt").unwrap().value_label_name, "grp2");
    assert_eq!(ds.datalabel, "New data label");
    let out = s.run("la li grp");
    assert!(out.contains("one") && out.contains("two"), "{out}");
    s.run("la l grp2");
    s.run("lab drop grp2");
    assert!(!s.ds().value_labels.contains_key("grp2"));
    s.fails("l var x \"no\"");
}

#[test]
fn macro_function_abbreviations() {
    let mut s = Session::with_fixture();
    s.run("format x %6.2f");
    s.run("local a : var la d");
    s.run("local b : variable l d");
    s.run("local c : val la d");
    s.run("local t : ty x");
    s.run("local f : f x");
    s.run("local g : form x");
    s.run("local h : lab yesno 1");
    s.run("local i : data l");
    let out = s.run("display \"`a'|`b'|`c'|`t'|`f'|`g'|`h'|`i'\"");
    assert_eq!(out.trim(), "Binary outcome|Binary outcome|yesno|double|%6.2f|%6.2f|yes|Ferrostat test fixture");
}

#[test]
fn clonevar_copies_metadata_and_respects_if() {
    let mut s = Session::new();
    let mut ds = fixture();
    ds.characteristics.entry("d".into()).or_default().insert("note".into(), "hello".into());
    s.load(ds);
    s.run("format d %4.0f");
    s.run("clonevar d2 = d");
    let src = s.ds().get_var("d").unwrap().clone();
    let v = s.ds().get_var("d2").unwrap().clone();
    assert_eq!(v.var_type, src.var_type);
    assert_eq!(v.format, "%4.0f");
    assert_eq!(v.label, "Binary outcome");
    assert_eq!(v.value_label_name, "yesno");
    assert_eq!(s.nums("d2"), s.nums("d"));
    let out = s.run("display \"`: char d2[note]'\"");
    assert_eq!(out.trim(), "hello");

    let out = s.run("clonevar x2 = x if id <= 4");
    assert!(out.contains("8 missing values generated"), "{out}");
    let x2 = s.nums("x2");
    let x = s.nums("x");
    assert_eq!(&x2[..4], &x[..4]);
    assert!(x2[4..].iter().all(|v| v.is_nan() || *v > 8.9e307));
    s.run("clonevar n2 = name in 1/2");
    let n2 = s.strs("n2");
    assert_eq!(n2[0], s.strs("name")[0]);
    assert_eq!(n2[5], "");
    s.fails("clonevar x2 = y");
    s.fails("clonevar q = nosuchvar");
}

#[test]
fn browse_and_edit_are_silent_in_batch() {
    let mut s = Session::with_fixture();
    for c in ["browse", "br", "br x y if id > 3", "edit", "ed x in 1/3", "bro"] {
        assert_eq!(s.run(c), "", "{c}");
    }
    s.fails("browse nosuchvar");
}

#[test]
fn describe_option_abbreviations() {
    let mut s = Session::with_fixture();
    let short = s.run("d, s");
    assert!(short.contains("Observations") && !short.contains("Storage"), "{short}");
    let simple = s.run("describe, si");
    assert_eq!(simple.trim(), "id  name  group  x  y  z  d  cnt");
    s.run("gen a_very_long_variable_name = 1");
    let full = s.run("describe a_very_long_variable_name, f");
    assert!(full.contains("a_very_long_variable_name"), "{full}");
    let abbr = s.run("describe a_very_long_variable_name");
    assert!(abbr.contains('~'), "{abbr}");
    let num = s.run("des x, n");
    assert!(num.lines().any(|l| l.trim_start().starts_with("4  x")), "{num}");
    assert_eq!(s.run("d, varl"), "");
}
