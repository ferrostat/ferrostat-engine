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

/// A workbook laid out the way spreadsheet programs write them: shared
/// strings, a custom date format and deflate compression.
fn excel_like_workbook(path: &std::path::Path) {
    let part = |s: &str| s.as_bytes().to_vec();
    let files = vec![
        ("[Content_Types].xml".to_string(), part(r#"<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>"#)),
        ("xl/workbook.xml".to_string(), part(r#"<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Notes" sheetId="1" r:id="rId1"/><sheet name="Data" sheetId="2" r:id="rId2"/></sheets></workbook>"#)),
        ("xl/_rels/workbook.xml.rels".to_string(), part(r#"<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="worksheet" Target="/xl/worksheets/sheet2.xml"/></Relationships>"#)),
        ("xl/sharedStrings.xml".to_string(), part(r#"<?xml version="1.0"?><sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="6" uniqueCount="6"><si><t>Province</t></si><si><t>Outflow (k)</t></si><si><t>Date</t></si><si><t>Forlì</t></si><si><r><t>Reggio </t></r><r><t>Emilia</t></r></si><si><t>notes &amp; comments</t></si></sst>"#)),
        ("xl/styles.xml".to_string(), part(r#"<?xml version="1.0"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><numFmts count="1"><numFmt numFmtId="164" formatCode="dd/mm/yyyy"/></numFmts><cellXfs count="2"><xf numFmtId="0"/><xf numFmtId="164" applyNumberFormat="1"/></cellXfs></styleSheet>"#)),
        ("xl/worksheets/sheet1.xml".to_string(), part(r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="s"><v>5</v></c></row></sheetData></worksheet>"#)),
        ("xl/worksheets/sheet2.xml".to_string(), part(r#"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>2</v></c></row>
<row r="2"><c r="A2" t="s"><v>3</v></c><c r="B2"><v>1046</v></c><c r="C2" s="1"><v>36526</v></c></row>
<row r="3"><c r="A3" t="s"><v>4</v></c><c r="B3"><v>2.5</v></c><c r="C3" s="1"><v>36527</v></c></row>
<row r="5"><c r="A5" t="inlineStr"><is><t>Bari</t></is></c><c r="C5" s="1"><v>36528</v></c></row>
</sheetData></worksheet>"#)),
    ];
    std::fs::write(path, ferrostat_cmd::xlsx::zip(&files)).unwrap();
}

#[test]
fn import_excel_sheet_firstrow_dates() {
    let dir = temp_dir("xlsx_import");
    let path = dir.join("book.xlsx");
    excel_like_workbook(&path);
    let mut s = Session::new();
    let out = s.run(&format!("import excel \"{}\", describe", path.display()));
    assert!(out.contains("Data | A1:C5"), "{out}");
    s.run(&format!("import excel \"{}\", sheet(\"Data\") firstrow clear", path.display()));
    assert_eq!(s.varnames(), ["Province", "Outflowk", "Date"]);
    assert_eq!(s.n(), 4);
    assert_eq!(s.strs("Province"), ["Forlì", "Reggio Emilia", "", "Bari"]);
    let out = s.nums("Outflowk");
    assert_eq!(&out[..2], [1046.0, 2.5]);
    assert!(out[2].is_nan() && out[3].is_nan());
    // spreadsheet dates become %td dates
    assert_eq!(s.nums("Date")[0], 14610.0);
    assert_eq!(s.ds().get_var("Date").unwrap().format, "%td");
    assert_eq!(s.ds().get_var("Outflowk").unwrap().label, "Outflow (k)");
}

#[test]
fn import_excel_defaults_and_cellrange() {
    let dir = temp_dir("xlsx_range");
    let path = dir.join("book.xlsx");
    excel_like_workbook(&path);
    let mut s = Session::new();
    // first sheet, no header row: variables are named after the columns
    s.run(&format!("import excel \"{}\", clear", path.display()));
    assert_eq!(s.varnames(), ["A"]);
    assert_eq!(s.strs("A"), ["notes & comments"]);
    s.run(&format!("import excel \"{}\", sheet(\"Data\") cellrange(A2:B3) clear", path.display()));
    assert_eq!(s.varnames(), ["A", "B"]);
    assert_eq!(s.nums("B"), [1046.0, 2.5]);
    s.run(&format!("import excel \"{}\", sheet(\"Data\") firstrow allstring case(lower) clear", path.display()));
    assert_eq!(s.strs("outflowk")[0], "1046");
}

#[test]
fn export_excel_roundtrip() {
    let dir = temp_dir("xlsx_export");
    let path = dir.join("out.xlsx");
    let mut s = Session::with_fixture();
    s.run("generate double when = mdy(1, 1, 2000) + _n");
    s.run("format when %td");
    s.run(&format!("export excel id name y when using \"{}\", firstrow(variables) replace", path.display()));
    s.run(&format!("export excel id using \"{}\" if group == 3, sheet(\"Third\", modify) firstrow(variables)", path.display()));
    s.run(&format!("import excel \"{}\", firstrow clear", path.display()));
    assert_eq!(s.n(), 12);
    assert_eq!(s.nums("y"), Y);
    assert_eq!(s.strs("name"), NAMES);
    assert_eq!(s.nums("when")[0], 14611.0);
    s.run(&format!("import excel \"{}\", sheet(\"Third\") firstrow clear", path.display()));
    assert_eq!(s.nums("id"), [9.0, 10.0, 11.0, 12.0]);
}
