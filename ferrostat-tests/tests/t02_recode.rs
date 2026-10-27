//! `recode` with value-label rules, `mvdecode`/`mvencode` with several
//! `\`-separated rules, and the duration/birthday date functions.

use ferrostat_tests::*;

fn ten() -> Session {
    let mut s = Session::new();
    s.run("set obs 10");
    s.run("gen x = _n");
    s
}

#[test]
fn recode_bare_and_quoted_labels_with_generate() {
    let mut s = ten();
    s.run("replace x = 1 in 1/4");
    s.run("replace x = 2 in 5/8");
    s.run("replace x = 3 in 9/10");
    let out = s.run("recode x (1=2 NE) (2=1 \"N Cntrl\"), gen(xn)");
    assert!(out.contains("8 differences between x and xn"), "{out}");
    assert_eq!(s.nums("xn"), vec![2., 2., 2., 2., 1., 1., 1., 1., 3., 3.]);
    let ds = s.ds();
    let v = ds.get_var("xn").unwrap();
    assert_eq!(v.value_label_name, "xn");
    assert_eq!(v.label, "RECODE of x");
    let vl = &ds.value_labels["xn"];
    assert_eq!(vl.lookup(2.0), Some("NE"));
    assert_eq!(vl.lookup(1.0), Some("N Cntrl"));
    // the source variable is untouched
    assert_eq!(s.nums("x")[0], 1.0);
}

#[test]
fn recode_ranges_first_rule_wins_and_label_option() {
    let mut s = ten();
    s.run("replace x = x * 1e6");
    s.run("recode x (0/2e+6=0 Small) (2e+6/4e+6=1 Medium) (4e+6/max=2 Large), gen(cx) label(sizes)");
    assert_eq!(s.nums("cx"), vec![0., 0., 1., 1., 2., 2., 2., 2., 2., 2.]);
    assert_eq!(s.ds().get_var("cx").unwrap().value_label_name, "sizes");
    assert!(s.ds().value_labels.contains_key("sizes"));
}

#[test]
fn recode_else_missing_min_max_copy() {
    let mut s = ten();
    s.run("replace x = . in 10");
    s.run("replace x = .a in 9");
    s.run("recode x (min=-1) (max=99) (missing=-9) (else=0), gen(a)");
    assert_eq!(s.nums("a"), vec![-1., 0., 0., 0., 0., 0., 0., 99., -9., -9.]);
    s.run("recode x (1/3=1) (nonmissing=copy) (.a=50), gen(b)");
    let b = s.nums("b");
    assert_eq!(&b[..9], &[1., 1., 1., 4., 5., 6., 7., 8., 50.]);
    assert!(b[9].is_nan());
    s.run("recode x (1/5=0) (else=1) in 1/6, gen(c) copyrest");
    let c = s.nums("c");
    assert_eq!(&c[..8], &[0., 0., 0., 0., 0., 1., 7., 8.]);
    // in place
    let out = s.run("recode x (1 2=10) (3=.)");
    assert!(out.contains("(x: 3 changes made)"), "{out}");
    let x = s.nums("x");
    assert_eq!((x[0], x[1], x[3]), (10., 10., 4.));
    assert!(x[2].is_nan());
}

#[test]
fn mvdecode_and_mvencode_multiple_rules() {
    let mut s = ten();
    s.run("replace x = 997 in 1/2");
    s.run("replace x = 999 in 3");
    s.run("replace x = 888 in 4");
    let out = s.run("mvdecode x, mv(997 999=.d \\ 888=.r)");
    assert!(out.contains("4 missing values generated"), "{out}");
    assert_eq!(s.scalar("x[1] == .d & x[3] == .d & x[4] == .r"), 1.0);
    s.run("mvencode x, mv(.r=888)");
    assert_eq!(s.nums("x")[3], 888.0);
    s.run("mvencode x in 1/2, mv(.d=997)");
    s.run("mvencode x in 3, mv(.d=999)");
    assert_eq!(&s.nums("x")[..4], &[997., 997., 999., 888.]);
    // value already present: refused without override
    s.run("replace x = .a in 5");
    s.run("replace x = .b in 6");
    s.run("replace x = . in 7");
    s.fails("mvencode x, mv(.a=888)");
    s.run("mvencode x, mv(.a=-1 \\ .b=-2 \\ else=-9) override");
    assert_eq!(&s.nums("x")[4..7], &[-1., -2., -9.]);
    // ranges in mvdecode
    s.run("mvdecode x, mv(-9/-1)");
    assert_eq!(s.scalar("missing(x[5]) & missing(x[6]) & missing(x[7])"), 1.0);
}

#[test]
fn age_and_birthdays() {
    let mut s = Session::new();
    assert_eq!(s.scalar("age(mdy(11,20,2011), mdy(4,15,2021))"), 9.0);
    assert_eq!(s.scalar("age(td(20nov2011), td(20nov2021))"), 10.0);
    assert_eq!(s.scalar("missing(age(td(20nov2011), td(19nov2011)))"), 1.0);
    assert_eq!(s.scalar("age(td(29feb2000), td(28feb2001))"), 0.0);
    assert_eq!(s.scalar("age(td(29feb2000), td(28feb2001), \"28feb\")"), 1.0);
    assert_close(s.scalar("age_frac(td(01jan2000), td(01jul2000))"), 182.0 / 366.0, 1e-12);
    assert_eq!(s.scalar("birthday(td(29feb2000), 2001)"), s.scalar("td(01mar2001)"));
    assert_eq!(s.scalar("birthday(td(29feb2000), 2001, \"28feb\")"), s.scalar("td(28feb2001)"));
    assert_eq!(s.scalar("previousbirthday(td(15jun1990), td(15jun2020))"), s.scalar("td(15jun2019)"));
    assert_eq!(s.scalar("nextbirthday(td(15jun1990), td(15jun2020))"), s.scalar("td(15jun2021)"));
    assert_eq!(s.scalar("nextbirthday(td(15jun1990), td(01jan2020))"), s.scalar("td(15jun2020)"));
}

#[test]
fn datediff_units() {
    let mut s = Session::new();
    assert_eq!(s.scalar("datediff(td(01jan2020), td(15mar2020), \"d\")"), 74.0);
    assert_eq!(s.scalar("datediff(td(15jan2020), td(14mar2020), \"m\")"), 1.0);
    assert_eq!(s.scalar("datediff(td(15jan2020), td(15mar2020), \"month\")"), 2.0);
    assert_eq!(s.scalar("datediff(td(15mar2020), td(15jan2020), \"m\")"), -2.0);
    assert_eq!(s.scalar("datediff(td(31jan2020), td(29feb2020), \"m\")"), 0.0);
    assert_eq!(s.scalar("datediff(td(01jan2000), td(31dec2020), \"y\")"), 20.0);
    assert_close(s.scalar("datediff_frac(td(01jan2020), td(16jan2020), \"m\")"), 15.0 / 31.0, 1e-12);
    assert_close(s.scalar("datediff_frac(td(01jan2001), td(01jul2001), \"y\")"), 181.0 / 365.0, 1e-12);
    s.fails("display datediff(td(01jan2020), td(16jan2020), \"w\")");
}

#[test]
fn calendar_helpers() {
    let mut s = Session::new();
    assert_eq!(s.scalar("isleapyear(2000) + isleapyear(1900) + isleapyear(2024)"), 2.0);
    assert_eq!(s.scalar("daysinmonth(td(10feb2024))"), 29.0);
    assert_eq!(s.scalar("firstdayofmonth(td(10feb2024))"), s.scalar("td(01feb2024)"));
    assert_eq!(s.scalar("lastdayofmonth(td(10feb2023))"), s.scalar("td(28feb2023)"));
    // 26oct2026 is a Monday
    assert_eq!(s.scalar("nextdow(td(26oct2026), 1)"), s.scalar("td(02nov2026)"));
    assert_eq!(s.scalar("previousdow(td(26oct2026), 0)"), s.scalar("td(25oct2026)"));
    assert_eq!(s.scalar("previousweekday(td(26oct2026))"), s.scalar("td(23oct2026)"));
    assert_eq!(s.scalar("nextleapyear(2024)"), 2028.0);
    assert_eq!(s.scalar("previousleapyear(2001)"), 2000.0);
}

#[test]
fn tutorial_age_line() {
    let mut s = Session::new();
    s.script(
        "set obs 3
gen date = date(\"15. April 2021\",\"DMY\")
gen birth = date(\"20. November 2011\",\"DMY\")
gen age = age(birth,date)",
    );
    assert_eq!(s.nums("age"), vec![9.0, 9.0, 9.0]);
}
