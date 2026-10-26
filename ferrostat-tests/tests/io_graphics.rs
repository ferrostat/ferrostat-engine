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

#[test]
fn date_axis_ticks_use_the_variable_format() {
    let dir = temp_dir("date_axis");
    let mut s = Session::new();
    s.run("set obs 15");
    s.run("generate date = mdy(1, 1, 2001) + _n");
    s.run("format date %td");
    s.run("generate double y = 1300 + 3 * _n");
    s.run("generate double z = 1290 + 2 * _n");
    s.run("twoway area y z date");
    let f = dir.join("g.svg");
    s.run(&format!("graph export \"{}\", replace", f.display()));
    let svg = std::fs::read_to_string(&f).unwrap();
    assert!(svg.contains("jan2001"), "x ticks should be dates");
    assert!(!svg.contains(">14980<") && !svg.contains(".0<"), "no raw day numbers or trailing .0");
    // the format survives graph save / graph use
    let gph = dir.join("g.gph");
    s.run(&format!("graph save \"{}\", replace", gph.display()));
    s.run(&format!("graph use \"{}\"", gph.display()));
    s.run(&format!("graph export \"{}\", replace", f.display()));
    assert!(std::fs::read_to_string(&f).unwrap().contains("jan2001"));
}

#[test]
fn float_cells_are_shown_at_float_precision() {
    let mut s = Session::new();
    s.run("set obs 2");
    s.run("generate float p = 1347.56");
    s.run("generate double d = 1347.56005859375");
    let out = s.run("list p d");
    assert!(out.contains(" 1347.56 "), "{out}");
    // a double keeps its digits
    assert!(out.contains("1347.5601"), "{out}");
}

#[test]
fn new_graph_types_survive_graph_save_and_use() {
    let dir = temp_dir("gph_roundtrip");
    let mut s = Session::new();
    s.script("set obs 40\ngenerate byte g = mod(_n, 4) + 1\nlabel define gl 1 \"one\" 2 \"two\" 3 \"three\" 4 \"four\"\nlabel values g gl\ngenerate double y = _n");
    for (cmd, marker) in [
        ("graph pie y, over(g) pie(2, explode) plabel(_all percent)", "pie\t"),
        ("graph dot y, over(g)", "guides\t"),
        ("graph hbar y, over(g)", "ylabel\t"),
        // layer fields: ... hollow, alpha, median marker, symbol, pattern, colour
        ("graph box y, over(g) medtype(marker)", "\tfalse\t0\ttrue\t"),
        ("twoway line y g, legend(order(1 \"Values\"))", "order\t1\tValues"),
    ] {
        s.run(cmd);
        let f = dir.join("g.gph");
        s.run(&format!("graph save \"{}\", replace", f.display()));
        let text = std::fs::read_to_string(&f).unwrap();
        assert!(text.contains(marker), "{cmd}: {marker:?} missing in\n{text}");
        let before = s.run(&format!("graph export \"{}\", replace", dir.join("a.svg").display()));
        s.run(&format!("graph use \"{}\"", f.display()));
        s.run(&format!("graph export \"{}\", replace", dir.join("b.svg").display()));
        let (a, b) = (std::fs::read_to_string(dir.join("a.svg")).unwrap(), std::fs::read_to_string(dir.join("b.svg")).unwrap());
        assert_eq!(a, b, "{cmd}: graph use draws a different graph");
        let _ = before;
    }
}

#[test]
fn pie_slices_are_group_sums_and_scheme_is_okabe_ito() {
    let dir = temp_dir("pie_scheme");
    let mut s = Session::new();
    s.script("set obs 4\ngenerate byte g = _n\ngenerate double v = _n");
    s.run("graph pie v, over(g) plabel(_all percent)");
    let f = dir.join("p.gph");
    s.run(&format!("graph save \"{}\", replace", f.display()));
    let text = std::fs::read_to_string(&f).unwrap();
    // 1+2+3+4 = 10: the fourth slice is 40%
    assert!(text.contains("pie\t4\tfalse\t4\t40%"), "{text}");
    // default palette: Okabe-Ito blue first; `set scheme classic` restores the old colours
    s.run("twoway line v g");
    let svg = dir.join("l.svg");
    s.run(&format!("graph export \"{}\", replace", svg.display()));
    assert!(std::fs::read_to_string(&svg).unwrap().to_uppercase().contains("#0072B2"));
    s.run("set scheme classic");
    s.run("twoway line v g");
    s.run(&format!("graph export \"{}\", replace", svg.display()));
    assert!(std::fs::read_to_string(&svg).unwrap().to_uppercase().contains("#1A476F"));
}

#[test]
fn graph_management_commands() {
    let dir = temp_dir("graph_mgmt");
    let mut s = Session::new();
    s.script("set obs 10\ngenerate x = _n\ngenerate y = x^2");
    s.run("scatter y x, name(first)");
    s.run("graph copy first second");
    s.run("graph rename second third");
    let dir_out = s.run("graph dir");
    assert!(dir_out.contains("first") && dir_out.contains("third") && !dir_out.contains("second"), "{dir_out}");
    s.fails("graph copy first third");
    s.run("graph copy first third, replace");
    let d = s.run("graph describe third");
    assert!(d.contains("scatter y x, name(first)") && d.contains("scheme:  okabeito"), "{d}");
    // describe a saved file too
    let f = dir.join("g.gph");
    s.run(&format!("graph save first \"{}\", replace", f.display()));
    assert!(s.run(&format!("graph describe \"{}\"", f.display())).contains("command:  scatter y x"));
    assert!(s.run("graph query, schemes").contains("okabeito"));
    assert!(s.run("graph query colorstyle").contains("navy"));
    assert!(s.run(&format!("graph replay first \"{}\"", f.display())).contains("(graph first displayed)"));
    // graph display with another scheme recolours the drawing
    s.run("graph display first, scheme(mono)");
    let svg = dir.join("m.svg");
    s.run(&format!("graph export \"{}\", replace", svg.display()));
    assert!(std::fs::read_to_string(&svg).unwrap().to_uppercase().contains("#282828"));
    // graph set svg fontface changes the exported font
    s.run("graph set svg fontface \"DejaVu Serif\"");
    s.run(&format!("graph export \"{}\", replace", svg.display()));
    assert!(std::fs::read_to_string(&svg).unwrap().contains("font-family=\"DejaVu Serif\""));
    assert!(s.run("graph set svg").contains("fontface"));
    // graph play applies a recording of option edits
    let rec = dir.join("edits.grec");
    std::fs::write(&rec, "* recording\ntitle(\"Edited title\")\nlegend(off)\n").unwrap();
    s.run(&format!("graph play \"{}\"", rec.display()));
    s.run(&format!("graph export \"{}\", replace", svg.display()));
    assert!(std::fs::read_to_string(&svg).unwrap().contains("Edited title"));
    // graph close / set graphics are accepted (front ends hide the graph window)
    s.run("graph close");
    s.run("set graphics off");
    s.run("set graphics on");
    s.run("set printcolor gs3");
    s.fails("set printcolor purple");
}

#[cfg(unix)]
#[test]
fn graph_print_uses_the_spooler() {
    let mut s = Session::new();
    s.script("set obs 5\ngenerate x = _n");
    s.run("scatter x x");
    // `true` stands in for lp: it accepts the file and succeeds
    s.run("graph set print command true");
    assert!(s.run("graph print").contains("sent to the printer"));
    s.run("graph set print command false");
    s.fails("graph print");
}

#[test]
fn marker_symbols_line_patterns_colours_and_palettes() {
    let dir = temp_dir("styles");
    let mut s = Session::new();
    s.script("set obs 20\ngenerate x = _n\ngenerate y = sin(_n)");
    let svg = dir.join("s.svg");
    s.run("twoway (scatter y x, msymbol(D) mcolor(red)) (line y x, lpattern(dash) lcolor(\"0 128 0\"))");
    s.run(&format!("graph export \"{}\", replace", svg.display()));
    let text = std::fs::read_to_string(&svg).unwrap().to_uppercase();
    assert!(text.contains("#FF0000") && text.contains("#008000"));
    assert!(text.contains("<POLYGON"), "diamonds are polygons");
    // a dashed line is drawn as many short segments
    assert!(text.matches("#008000").count() > 10);
    s.fails("scatter y x, mcolor(notacolour)");
    for p in ["palette color navy \"255 128 0\"", "palette linepalette", "palette symbolpalette"] {
        s.run(p);
        s.run(&format!("graph export \"{}\", replace", svg.display()));
    }
    assert!(s.run("palette color navy").contains("navy: 0 0 128"));
}

/// Layers of the current graph as saved in a .gph file: (kind, points, ranges).
fn saved_layers(s: &mut Session, dir: &std::path::Path) -> Vec<(String, Vec<(f64, f64)>, Vec<(f64, f64, f64)>)> {
    let f = dir.join("layers.gph");
    s.run(&format!("graph save \"{}\", replace", f.display()));
    let mut out: Vec<(String, Vec<(f64, f64)>, Vec<(f64, f64, f64)>)> = Vec::new();
    for line in std::fs::read_to_string(&f).unwrap().lines() {
        let v: Vec<&str> = line.split('\t').collect();
        match v[0] {
            "layer" => out.push((v[1].to_string(), Vec::new(), Vec::new())),
            "p" => out.last_mut().unwrap().1.push((v[1].parse().unwrap(), v[2].parse().unwrap())),
            "r" => out.last_mut().unwrap().2.push((v[1].parse().unwrap(), v[2].parse().unwrap(), v[3].parse().unwrap())),
            _ => {}
        }
    }
    out
}

#[test]
fn twoway_function_histogram_kdensity() {
    let dir = temp_dir("tw_fn_hist");
    let mut s = Session::new();
    s.run("twoway function y = x^2, range(0 2) n(3) droplines(1)");
    let l = saved_layers(&mut s, &dir);
    assert_eq!(l[0].1, vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)]);
    assert_eq!(l[1].2, vec![(1.0, 0.0, 1.0)], "dropline from 0 to f(1)");
    s.run("twoway function normalden(x), range(-1 1) n(2) horizontal");
    let l = saved_layers(&mut s, &dir);
    assert!((l[0].1[0].0 - 0.24197072451914337).abs() < 1e-12 && l[0].1[0].1 == -1.0, "horizontal swaps axes");
    s.script("clear\nset obs 10\ngenerate v = _n");
    s.run("twoway histogram v, bin(2) frequency");
    let l = saved_layers(&mut s, &dir);
    assert_eq!(l[0].2.iter().map(|r| r.2).collect::<Vec<_>>(), vec![5.0, 5.0]);
    s.run("twoway histogram v, discrete percent horizontal");
    let l = saved_layers(&mut s, &dir);
    assert_eq!(l[0].2.len(), 10);
    assert!(l[0].2.iter().all(|r| (r.2 - 10.0).abs() < 1e-12));
    // a density integrates to about one
    s.run("twoway kdensity v, n(400) range(-10 21)");
    let l = saved_layers(&mut s, &dir);
    let area: f64 = l[0].1.windows(2).map(|w| (w[1].0 - w[0].0) * (w[1].1 + w[0].1) / 2.0).sum();
    assert!((area - 1.0).abs() < 1e-3, "{area}");
}

#[test]
fn twoway_smoothers_and_dot_and_tsline() {
    let dir = temp_dir("tw_smooth");
    let mut s = Session::new();
    s.script("set obs 40\ngenerate x = _n\ngenerate y = 3 + 2*x\ngenerate t = _n\ntsset t");
    s.run("twoway lowess y x");
    for (x, y) in &saved_layers(&mut s, &dir)[0].1 {
        assert!((y - (3.0 + 2.0 * x)).abs() < 1e-9);
    }
    s.run("twoway lpolyci y x, degree(1)");
    let l = saved_layers(&mut s, &dir);
    assert!(l[0].0.contains("Area") && l[1].0 == "Line");
    for ((x, m), (_, lo, hi)) in l[1].1.iter().zip(&l[0].2) {
        assert!((m - (3.0 + 2.0 * x)).abs() < 1e-9 && lo <= m && m <= hi);
    }
    s.run("twoway mband y x, bands(4)");
    assert_eq!(saved_layers(&mut s, &dir)[0].1.len(), 4);
    s.run("twoway mspline y x, bands(4)");
    for (x, y) in &saved_layers(&mut s, &dir)[0].1 {
        assert!((y - (3.0 + 2.0 * x)).abs() < 1e-9, "a spline through collinear medians is that line");
    }
    s.run("twoway dot y x in 1/5");
    assert_eq!(saved_layers(&mut s, &dir)[0].0, "Dot");
    // twoway tsline plots against the tsset time variable; qualifiers stop at ||
    s.run("twoway tsline y in 1/5 || tsline x in 1/5");
    let l = saved_layers(&mut s, &dir);
    assert_eq!(l.len(), 2);
    assert_eq!(l[0].1.iter().map(|p| p.0).collect::<Vec<_>>(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn paired_coordinate_immediate_and_range_point_plots() {
    let dir = temp_dir("tw_pc");
    let mut s = Session::new();
    s.script("set obs 3\ngenerate a = _n\ngenerate b = 2*_n\ngenerate c = _n + 1\ngenerate d = 2*_n + 1");
    for verb in ["pcspike", "pccapsym", "pcarrow", "pcbarrow", "pcscatter"] {
        s.run(&format!("twoway {verb} a b c d"));
        let l = saved_layers(&mut s, &dir);
        assert!(l[0].0.starts_with("Pc("), "{verb}: {}", l[0].0);
    }
    // segments are stored as (y1, x1, y2, x2)
    let f = dir.join("pc.gph");
    s.run("twoway pcarrow a b c d");
    s.run(&format!("graph save \"{}\", replace", f.display()));
    assert!(std::fs::read_to_string(&f).unwrap().contains("g\t1 2 2 3\n"));
    s.fails("twoway pcspike a b c");
    // immediate forms
    s.run("twoway scatteri 1 2 \"first\" 3 4 (9) \"second\"");
    let l = saved_layers(&mut s, &dir);
    assert_eq!(l[0].1, vec![(2.0, 1.0), (4.0, 3.0)]);
    s.run(&format!("graph save \"{}\", replace", f.display()));
    let text = std::fs::read_to_string(&f).unwrap();
    assert!(text.contains("m\tfirst") && text.contains("m\tsecond"));
    s.run("twoway pci 0 0 1 1 1 0 0 1 || pcarrowi 0.5 0 0.5 1");
    assert_eq!(saved_layers(&mut s, &dir).len(), 2);
    s.fails("twoway scatteri 1 2 3");
    // range with a point: y1 y2 y3 x
    s.run("twoway rpcap c a b a");
    let l = saved_layers(&mut s, &dir);
    assert!(l[0].0.contains("PointCap"));
    assert_eq!(l[0].2[0], (1.0, 2.0, 1.0));
    assert_eq!(l[0].1[0], (1.0, 2.0));
    s.run("twoway rpspike c a b a, horizontal");
    assert!(saved_layers(&mut s, &dir)[0].0.contains("PointSpike"));
}

#[test]
fn contour_heatmap_and_graph_matrix() {
    let dir = temp_dir("surfaces");
    let mut s = Session::new();
    // a complete 5x5 grid of z = x + y
    s.script("set obs 25\ngenerate x = mod(_n-1, 5)\ngenerate y = floor((_n-1)/5)\ngenerate z = x + y");
    let f = dir.join("s.gph");
    s.run("twoway heatmap z y x, levels(4)");
    s.run(&format!("graph save \"{}\", replace", f.display()));
    let text = std::fs::read_to_string(&f).unwrap();
    assert_eq!(text.matches("cell\t").count(), 25, "one cell per observation");
    assert_eq!(text.matches("ckey\t").count(), 4, "four colour bands");
    s.run("twoway contour z y x, ccuts(2 4 6)");
    s.run(&format!("graph save \"{}\", replace", f.display()));
    let text = std::fs::read_to_string(&f).unwrap();
    assert_eq!(text.matches("cell\t").count(), 16, "a 5x5 grid has 4x4 cells");
    assert!(text.contains("ckey\t0 - 2\t") && text.contains("ckey\t6 - 8\t"), "{text}");
    assert!(text.contains("ytitle\ty"), "the y axis is y, not z");
    // contour lines of the plane x + y = 4 run along the anti-diagonal
    s.run("twoway contourline z y x, ccuts(4)");
    s.run(&format!("graph save \"{}\", replace", f.display()));
    let segs: Vec<Vec<f64>> = std::fs::read_to_string(&f).unwrap().lines().filter(|l| l.starts_with("g\t")).map(|l| l.split('\t').nth(1).unwrap().split(' ').map(|v| v.parse().unwrap()).collect()).collect();
    assert!(!segs.is_empty());
    for g in segs {
        assert!((g[0] + g[1] - 4.0).abs() < 1e-9 && (g[2] + g[3] - 4.0).abs() < 1e-9, "{g:?}");
    }
    // scattered data are interpolated
    s.script("clear\nset obs 50\nset seed 4\ngenerate x = runiform()\ngenerate y = runiform()\ngenerate z = x - y");
    s.run("twoway contour z y x");
    s.run("twoway contour z y x, interp(shepard)");
    s.fails("twoway contour z y x, interp(none)");
    // scatterplot matrix: k x k panels, names on the diagonal
    s.run("graph matrix x y z, half");
    s.run(&format!("graph save \"{}\", replace", f.display()));
    let svg = dir.join("m.svg");
    s.run(&format!("graph export \"{}\", replace", svg.display()));
    // text content sits on its own line in the SVG output
    assert!(std::fs::read_to_string(&svg).unwrap().replace(['\n', ' '], "").contains(">x</text>"));
    s.fails("graph matrix x");
}
