//! `unique`, `frmttable` and `putexcel` cell formatting.

use ferrostat_cmd::xlsx::{unzip, Cell, Workbook};
use ferrostat_tests::*;

#[test]
fn unique_counts_values_and_records() {
    let mut s = Session::with_fixture();
    let out = s.run("unique group");
    assert!(out.contains("Number of unique values of group is  3"), "{out}");
    assert!(out.contains("Number of records is  12"), "{out}");
    assert_eq!(s.scalar("r(unique)"), 3.0);
    assert_eq!(s.scalar("r(N)"), 12.0);
    s.run("quietly unique group d if x > 2");
    assert_eq!(s.scalar("r(unique)"), 6.0);
    assert_eq!(s.scalar("r(N)"), 10.0);
}

#[test]
fn unique_by_generates_counts() {
    let mut s = Session::with_fixture();
    s.run("unique z, by(group) generate(nz)");
    // group 1: 3 1 4 1 -> 3; group 2: 5 9 2 6 -> 4; group 3: 5 3 5 8 -> 3
    assert_eq!(s.scalar("r(sum)"), 10.0);
    assert_eq!(s.scalar("r(max)"), 4.0);
    assert_eq!(s.scalar("r(unique)"), 8.0);
    let nz: Vec<f64> = s.nums("nz").into_iter().filter(|v| *v < 1e300).collect();
    assert_eq!(nz, vec![3.0, 4.0, 3.0]);
}

#[test]
fn frmttable_merge_substat_annotate() {
    let dir = temp_dir("frmttable");
    let path = dir.join("tab.rtf");
    let mut s = Session::new();
    s.run("matrix A = (52 \\ 22)");
    s.run("matrix rownames A = 0 1");
    s.run("matrix B = (6.6249, 1.2504 \\ 4.6052, 1.8146)");
    s.run("matrix S = (3, 0 \\ 2, 0)");
    let out = s.run("frmttable, statmat(A) sdec(0) ctitles(\"Foreign\",\"N\")");
    assert!(out.contains("Foreign") && out.contains("52"), "{out}");
    let out = s.run("frmttable, statmat(B) merge sdec(2) ctitles(\"Coefficient\") substat(1) annotate(S) asymbol(*,**,***)");
    assert!(out.contains("6.62***"), "{out}");
    assert!(out.contains("(1.25)"), "{out}");
    assert!(out.contains("4.61**"), "{out}");
    let lines: Vec<&str> = out.lines().collect();
    let row0 = lines.iter().find(|l| l.contains("6.62***")).unwrap();
    assert!(row0.contains("52"), "merged row should hold both columns: {out}");
    s.run(&format!("frmttable using \"{}\", statmat(B) merge sdec(1) ctitles(\"Again\") substat(1) replace", path.display()));
    let rtf = std::fs::read_to_string(&path).unwrap();
    assert!(rtf.starts_with("{\\rtf1"));
    assert!(rtf.contains("Coefficient") && rtf.contains("Again") && rtf.contains("6.62***") && rtf.contains("(1.3)"), "{rtf}");
    // without replace the file is not overwritten
    assert!(s.try_run(&format!("frmttable using \"{}\", statmat(A)", path.display())).is_err());
    let tex = dir.join("tab.tex");
    s.run(&format!("frmttable using \"{}\", statmat(A) sdec(0) rtitles(\"a\" \\ \"b\") title(\"My table\") tex replace", tex.display()));
    let t = std::fs::read_to_string(&tex).unwrap();
    assert!(t.contains("\\begin{tabular}") && t.contains("a & 52") && t.contains("My table"), "{t}");
}

#[test]
fn putexcel_formats_are_written_to_styles() {
    let dir = temp_dir("putexcel_fmt");
    let path = dir.join("fmt.xlsx");
    let mut s = Session::new();
    s.run("matrix d = (1234.5, 0.25 \\ 2000, 0.5)");
    s.run("matrix colnames d = N Share");
    s.run(&format!("putexcel set \"{}\", sheet(Sheet1) replace", path.display()));
    s.run("putexcel B1 = matrix(d), colnames");
    s.run("putexcel B1:C1, border(bottom, medium, black) bold");
    s.run("putexcel B2:B3, nformat(\"number_sep\")");
    s.run("putexcel C2:C3, nformat(percent) font(Arial, 12, red) hcenter fpattern(solid, lightgray)");
    s.run("putexcel E1:F1, merge txtwrap");
    s.run("putexcel E5 = 3.14159, nformat(\"0.000\") italic");
    let bytes = std::fs::read(&path).unwrap();
    let files = unzip(&bytes).unwrap();
    let styles = String::from_utf8_lossy(&files["xl/styles.xml"]).to_string();
    assert!(styles.contains(r#"<bottom style="medium"><color rgb="FF000000"/></bottom>"#), "{styles}");
    assert!(styles.contains(r#"numFmtId="3""#) && styles.contains(r#"numFmtId="9""#), "{styles}");
    assert!(styles.contains(r#"formatCode="0.000""#), "{styles}");
    assert!(styles.contains(r#"<name val="Arial"/>"#) && styles.contains(r#"rgb="FFFF0000""#), "{styles}");
    assert!(styles.contains(r#"patternType="solid""#) && styles.contains(r#"horizontal="center""#), "{styles}");
    let sheet = String::from_utf8_lossy(&files["xl/worksheets/sheet1.xml"]).to_string();
    assert!(sheet.contains(r#"<mergeCell ref="E1:F1"/>"#), "{sheet}");
    // read back through the workbook reader
    let book = Workbook::read(&bytes).unwrap();
    let st = &book.styles["Sheet1"];
    assert_eq!(st[&(1, 2)].borders[3].as_ref().map(|b| b.0.as_str()), Some("medium"));
    assert!(st[&(1, 2)].bold);
    assert_eq!(st[&(2, 2)].num_fmt.as_deref(), Some("#,##0"));
    assert_eq!(st[&(2, 3)].num_fmt.as_deref(), Some("0%"));
    assert_eq!(st[&(2, 3)].font_name.as_deref(), Some("Arial"));
    assert!(st[&(1, 5)].wrap);
    assert!(st[&(5, 5)].italic);
    assert_eq!(book.sheets[0].1[&(5, 5)], Cell::Num(3.14159));
    assert_eq!(book.sheets[0].1[&(2, 2)], Cell::Num(1234.5));
    // formats merge with existing ones unless overwritefmt is given
    s.run("putexcel B1:C1, italic");
    s.run("putexcel C2, overwritefmt nformat(number_d2)");
    let book = Workbook::read(&std::fs::read(&path).unwrap()).unwrap();
    let st = &book.styles["Sheet1"];
    assert!(st[&(1, 2)].bold && st[&(1, 2)].italic);
    assert_eq!(st[&(2, 3)].num_fmt.as_deref(), Some("0.00"));
    assert_eq!(st[&(2, 3)].font_name, None);
}

#[test]
fn putexcel_matrix_names_inside_parentheses() {
    let dir = temp_dir("putexcel_names");
    let path = dir.join("m.xlsx");
    let mut s = Session::new();
    s.run("matrix A = (1, 2 \\ 3, 4)");
    s.run("matrix rownames A = a b");
    s.run("matrix colnames A = x y");
    s.run(&format!("putexcel set \"{}\", replace", path.display()));
    s.run("putexcel A1 = mat(A, names)");
    let book = Workbook::read(&std::fs::read(&path).unwrap()).unwrap();
    let c = &book.sheets[0].1;
    assert_eq!(c[&(1, 2)], Cell::Str("x".into()));
    assert_eq!(c[&(2, 1)], Cell::Str("a".into()));
    assert_eq!(c[&(3, 3)], Cell::Num(4.0));
}
