//! Text import/export, logs, Excel output and graphics.

use ferrostat_tests::*;

#[test]
fn import_and_export_delimited_roundtrip() {
    let dir = temp_dir("csv");
    let path = dir.join("fx.csv");
    let mut s = Session::with_fixture();
    s.run(&format!("export delimited using \"{}\", replace", path.display()));
    s.run("clear");
    s.run(&format!("import delimited \"{}\", clear", path.display()));
    assert_eq!(s.n(), 12);
    assert_eq!(s.nums("y"), Y);
    assert_eq!(s.strs("name")[3], "dog");
}

#[test]
fn import_delimited_infers_types_and_missing() {
    let dir = temp_dir("csv_types");
    let path = dir.join("t.csv");
    std::fs::write(&path, "id,Score,label\n1,2.5,a\n2,,b\n3,4,c\n").unwrap();
    let mut s = Session::new();
    s.run(&format!("import delimited \"{}\", clear", path.display()));
    assert_eq!(s.varnames(), ["id", "score", "label"]);
    assert!(s.nums("score")[1].is_nan());
}

#[test]
fn log_file_is_written() {
    let dir = temp_dir("log");
    let path = dir.join("out.log");
    let mut s = Session::with_fixture();
    s.run(&format!("log using \"{}\", text replace", path.display()));
    s.run("summarize x");
    s.run("log close");
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("summarize x") && text.contains("Obs"), "{text}");
}

#[test]
fn putexcel_writes_xlsx() {
    let dir = temp_dir("xlsx");
    let path = dir.join("t.xlsx");
    let mut s = Session::new();
    s.run("matrix M = (1, 2 \\ 3, 4)");
    s.run(&format!("putexcel set \"{}\", sheet(\"Results\") replace", path.display()));
    s.run("putexcel A1 = \"Title\"");
    s.run("putexcel A3 = matrix(M), names");
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..2], b"PK");
}

#[test]
fn graph_export_png_and_svg() {
    let dir = temp_dir("graph");
    let mut s = Session::with_fixture();
    s.run("scatter y x");
    let png = dir.join("s.png");
    s.run(&format!("graph export \"{}\", replace", png.display()));
    assert!(std::fs::metadata(&png).unwrap().len() > 1000);
    s.run("histogram y, frequency");
    let svg = dir.join("h.svg");
    s.run(&format!("graph export \"{}\", replace", svg.display()));
    assert!(std::fs::read_to_string(&svg).unwrap().contains("<svg"));
}

#[test]
fn graphs_do_not_write_files_without_export() {
    let dir = temp_dir("nofiles");
    let before = std::fs::read_dir(".").unwrap().count();
    let mut s = Session::with_fixture();
    s.run("scatter y x");
    s.run("histogram y");
    assert_eq!(std::fs::read_dir(".").unwrap().count(), before);
    let _ = dir;
}

#[test]
fn import_delimited_with_italian_locale() {
    // decimal comma and dot as thousands separator, as written by Italian
    // spreadsheets (needed by test-andrea/DFA.do)
    let dir = temp_dir("csv_locale");
    let path = dir.join("it.csv");
    std::fs::write(&path, "year,outflow,wage\n1876,\"1.046\",\"1,16\"\n1877,999,\"2,11\"\n1878,\"12.345,5\",\n").unwrap();
    let mut s = Session::new();
    s.run(&format!("import delimited \"{}\", parselocale(it_IT) decimalseparator(,) clear", path.display()));
    assert_eq!(s.nums("outflow"), [1046.0, 999.0, 12345.5]);
    assert_eq!(&s.nums("wage")[..2], [1.16, 2.11]);
    assert!(s.nums("wage")[2].is_nan());
}

#[test]
fn import_delimited_decimal_comma_only() {
    let dir = temp_dir("csv_dec");
    let path = dir.join("d.csv");
    std::fs::write(&path, "a;b\n\"1,5\";2\n\"3,25\";4\n").unwrap();
    let mut s = Session::new();
    s.run(&format!("import delimited \"{}\", delimiters(\";\") decimalseparator(,) clear", path.display()));
    assert_eq!(s.nums("a"), [1.5, 3.25]);
}

#[test]
fn outreg2_builds_regression_table() {
    // outreg2 is used throughout test-andrea/Spatial.do
    let dir = temp_dir("outreg");
    let path = dir.join("t.xls");
    let mut s = Session::with_fixture();
    s.run("regress y x");
    s.run(&format!("outreg2 using \"{}\", excel replace title(\"My models\") stats(coef se pval) ctitle(\"M1\")", path.display()));
    s.run("regress y x z");
    s.run("scalar k = 42");
    s.run(&format!("outreg2 using \"{}\", excel append stats(coef se pval) addstat(\"Extra\", k) ctitle(\"M2\")", path.display()));
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.starts_with("My models"), "{text}");
    let header = text.lines().find(|l| l.starts_with("VARIABLES")).unwrap();
    assert_eq!(header, "VARIABLES\tM1\tM2");
    let zrow = text.lines().find(|l| l.starts_with("z\t")).unwrap();
    assert!(zrow.split('\t').nth(1).unwrap().is_empty(), "{zrow}");
    assert!(text.contains("Observations\t12\t12"), "{text}");
    assert!(text.contains("Extra\t\t42.000"), "{text}");
}
