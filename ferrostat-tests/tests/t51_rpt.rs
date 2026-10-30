//! Reporting wave 6: docx2txt/docx2html/docx2markdown/docx2epub/docx2pdf,
//! pdf2txt (own and Flate-compressed PDFs), putdocx/putpdf collect,
//! collect copy / style notes / style txt, putexcel formulas, pictures,
//! formats and open/save mode, set docx_hardbreak/docx_paramode and ml
//! with survey designs.

use ferrostat_tests::*;
use std::io::Write;

fn path(dir: &std::path::Path, f: &str) -> String {
    dir.join(f).to_string_lossy().replace('\\', "/")
}

fn read(p: &str) -> String {
    String::from_utf8_lossy(&std::fs::read(p).unwrap()).to_string()
}

fn regression_data(s: &mut Session) {
    s.script(
        "set obs 40
         set seed 11
         gen x = rnormal()
         gen z = rnormal()
         gen y = 1 + x - z + rnormal()",
    );
}

#[test]
fn putdocx_round_trips_through_converters() {
    let dir = temp_dir("t51_conv");
    let docx = path(&dir, "rep.docx");
    let mut s = Session::new();
    regression_data(&mut s);
    s.run("putdocx begin");
    s.run("putdocx paragraph, style(Title)");
    s.run("putdocx text (\"Quarterly summary\")");
    s.run("putdocx paragraph");
    s.run("putdocx text (\"Plain text and \")");
    s.run("putdocx text (\"bold words\"), bold");
    s.run("putdocx table t = (2,2)");
    s.run("putdocx table t(1,1) = (\"alpha\")");
    s.run("putdocx table t(2,2) = (42)");
    s.run(&format!("putdocx save \"{docx}\", replace"));

    s.run(&format!("docx2txt \"{docx}\""));
    let txt = read(&path(&dir, "rep.txt"));
    assert!(txt.contains("Quarterly summary"), "{txt}");
    assert!(txt.contains("Plain text and bold words"), "{txt}");
    assert!(txt.contains("alpha") && txt.contains("42"), "{txt}");
    // the target exists now
    s.fails(&format!("docx2txt \"{docx}\""));
    s.run(&format!("docx2txt \"{docx}\", replace"));

    let html = path(&dir, "out.html");
    s.run(&format!("docx2html \"{docx}\", saving(\"{html}\")"));
    let h = read(&html);
    assert!(h.contains("<h1>Quarterly summary</h1>"), "{h}");
    assert!(h.contains("<strong>bold words</strong>"), "{h}");
    assert!(h.contains("<td>alpha</td>"), "{h}");

    s.run(&format!("docx2markdown \"{docx}\""));
    let md = read(&path(&dir, "rep.md"));
    assert!(md.starts_with("# Quarterly summary"), "{md}");
    assert!(md.contains("**bold words**"), "{md}");
    assert!(md.contains("| alpha |"), "{md}");

    s.run(&format!("docx2epub \"{docx}\""));
    let epub = std::fs::read(path(&dir, "rep.epub")).unwrap();
    assert_eq!(&epub[30..38], b"mimetype");
    let parts = ferrostat_cmd::xlsx::unzip(&epub).unwrap();
    assert_eq!(parts["mimetype"], b"application/epub+zip");
    assert!(parts.contains_key("META-INF/container.xml"));
    let content = String::from_utf8_lossy(&parts["OEBPS/content.xhtml"]).to_string();
    assert!(content.contains("<h1 id=\"h1\">Quarterly summary</h1>"), "{content}");
    let nav = String::from_utf8_lossy(&parts["OEBPS/nav.xhtml"]).to_string();
    assert!(nav.contains("content.xhtml#h1"));

    s.run(&format!("docx2pdf \"{docx}\""));
    let pdf = path(&dir, "rep.pdf");
    assert!(std::fs::read(&pdf).unwrap().starts_with(b"%PDF"));
    s.run(&format!("pdf2txt \"{pdf}\", saving(\"{}\")", path(&dir, "frompdf.txt")));
    let back = read(&path(&dir, "frompdf.txt"));
    assert!(back.contains("Quarterly summary"), "{back}");
    assert!(back.contains("Plain text and bold words"), "{back}");
    assert!(back.contains("alpha"), "{back}");
    s.fails(&format!("docx2txt \"{}\"", path(&dir, "missing.docx")));
}

#[test]
fn docx_from_another_writer() {
    // a hand-written document: tabs, a line break, a page break, a merged cell
    let dir = temp_dir("t51_foreign");
    let doc = r#"<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Intro</w:t></w:r></w:p>
<w:p><w:r><w:rPr><w:i/></w:rPr><w:t xml:space="preserve">one</w:t></w:r><w:r><w:tab/><w:t>two</w:t><w:br/><w:t>three</w:t></w:r></w:p>
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
<w:tbl><w:tr><w:tc><w:tcPr><w:gridSpan w:val="2"/></w:tcPr><w:p><w:r><w:t>wide</w:t></w:r></w:p></w:tc></w:tr><w:tr><w:tc><w:p><w:r><w:t>a</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>b</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
<w:sectPr><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/></w:sectPr></w:body></w:document>"#;
    let bytes = ferrostat_cmd::xlsx::zip(&[("word/document.xml".to_string(), doc.as_bytes().to_vec())]);
    let src = path(&dir, "foreign.docx");
    std::fs::write(&src, bytes).unwrap();
    let mut s = Session::new();
    s.run(&format!("docx2txt \"{src}\""));
    let t = read(&path(&dir, "foreign.txt"));
    assert!(t.starts_with("Intro\none\ttwo\nthree\n"), "{t:?}");
    assert!(t.contains('\u{c}'));
    assert!(t.contains("wide") && t.lines().any(|l| l.split_whitespace().collect::<Vec<_>>() == ["a", "b"]), "{t}");
    s.run(&format!("docx2markdown \"{src}\""));
    let md = read(&path(&dir, "foreign.md"));
    assert!(md.contains("## Intro") && md.contains("*one*"), "{md}");
    s.run(&format!("docx2pdf \"{src}\""));
    let p = read(&path(&dir, "foreign.pdf"));
    assert!(p.contains("/MediaBox [0 0 842 595]"), "landscape A4 expected");
}

fn deflate(data: &[u8]) -> Vec<u8> {
    let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    e.write_all(data).unwrap();
    e.finish().unwrap()
}

#[test]
fn pdf2txt_flate_tounicode_and_object_streams() {
    let dir = temp_dir("t51_pdf");
    let content = deflate(b"BT /F1 10 Tf 1 0 0 1 72 720 Tm (Compressed) Tj 0 -14 TD [(word)-300(gap)] TJ T* (A) Tj ET BT /F2 10 Tf 72 600 Td <00010002> Tj ET");
    let cmap = deflate(b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap 1 begincodespacerange <0000> <FFFF> endcodespacerange 2 beginbfchar <0001> <0048> <0002> <0069> endbfchar endcmap end end");
    // the page tree and fonts live in a compressed object stream
    let objs = b"<< /Type /Pages /Kids [3 0 R] /Count 1 /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >> << /Type /Page /Parent 2 0 R /Contents 4 0 R >> << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> << /Type /Font /Subtype /Type0 /BaseFont /X /Encoding /Identity-H /ToUnicode 7 0 R >>";
    // offsets of the four objects inside `objs`
    let s = String::from_utf8_lossy(objs).to_string();
    let mut offs = vec![0usize];
    let mut depth = 0i32;
    let b = s.as_bytes();
    for i in 0..b.len() - 1 {
        if b[i] == b'<' && b[i + 1] == b'<' {
            depth += 1;
        }
        if b[i] == b'>' && b[i + 1] == b'>' {
            depth -= 1;
            if depth == 0 && i + 3 < b.len() {
                offs.push(i + 3);
            }
        }
    }
    let header = format!("2 {} 3 {} 5 {} 6 {} ", offs[0], offs[1], offs[2], offs[3]);
    let mut stm = header.clone().into_bytes();
    stm.extend_from_slice(objs);
    let stm = deflate(&stm);
    let mut pdf = b"%PDF-1.5\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n".to_vec();
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} /Filter /FlateDecode >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(&content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");
    pdf.extend_from_slice(format!("7 0 obj\n<< /Length 8 0 R /Filter [/FlateDecode] >>\nstream\n").as_bytes());
    pdf.extend_from_slice(&cmap);
    pdf.extend_from_slice(format!("\nendstream\nendobj\n8 0 obj\n{}\nendobj\n", cmap.len()).as_bytes());
    pdf.extend_from_slice(format!("9 0 obj\n<< /Type /ObjStm /N 4 /First {} /Length {} /Filter /FlateDecode >>\nstream\n", header.len(), stm.len()).as_bytes());
    pdf.extend_from_slice(&stm);
    pdf.extend_from_slice(b"\nendstream\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n");
    let src = path(&dir, "c.pdf");
    std::fs::write(&src, &pdf).unwrap();
    let mut ses = Session::new();
    ses.run(&format!("pdf2txt \"{src}\""));
    let t = read(&path(&dir, "c.txt"));
    assert_eq!(t, "Compressed\nword gap\nA\n\nHi\n", "{t:?}");
    std::fs::write(path(&dir, "bad.pdf"), b"not a pdf").unwrap();
    ses.fails(&format!("pdf2txt \"{}\"", path(&dir, "bad.pdf")));
}

#[test]
fn putpdf_text_extracts() {
    let dir = temp_dir("t51_putpdf");
    let pdf = path(&dir, "p.pdf");
    let mut s = Session::new();
    s.run("putpdf begin");
    s.run("putpdf paragraph");
    s.run("putpdf text (\"Some (parenthesised) text, caf\u{e9}\")");
    s.run("putpdf pagebreak");
    s.run("putpdf paragraph");
    s.run("putpdf text (\"Second page\")");
    s.run(&format!("putpdf save \"{pdf}\""));
    s.run(&format!("pdf2txt \"{pdf}\""));
    let t = read(&path(&dir, "p.txt"));
    assert_eq!(t, "Some (parenthesised) text, caf\u{e9}\n\u{c}Second page\n", "{t:?}");
}

#[test]
fn put_collect_tables() {
    let dir = temp_dir("t51_collect");
    let mut s = Session::new();
    regression_data(&mut s);
    s.run("collect clear");
    s.run("collect: regress y x z");
    s.run("collect layout (colname) (result[_r_b _r_se])");
    s.run("collect title \"Model one\"");
    s.run("collect notes \"Synthetic data\"");
    s.run("collect style notes, prefix(\"Note: \")");
    // putdocx collect needs an open document
    s.fails("putdocx collect");
    s.run("putdocx begin");
    s.run("putdocx collect, tablename(reg)");
    let d = path(&dir, "c.docx");
    s.run(&format!("putdocx save \"{d}\""));
    s.run(&format!("docx2txt \"{d}\""));
    let t = read(&path(&dir, "c.txt"));
    assert!(t.contains("Model one"), "{t}");
    assert!(t.contains("Coefficient") && t.contains("Std. error"), "{t}");
    assert!(t.lines().any(|l| l.starts_with("x ")), "{t}");
    assert!(t.contains("Note: Synthetic data"), "{t}");

    s.run("putpdf begin");
    s.run("putpdf collect");
    let p = path(&dir, "c.pdf");
    s.run(&format!("putpdf save \"{p}\""));
    s.run(&format!("pdf2txt \"{p}\", replace"));
    let t = read(&path(&dir, "c.txt"));
    assert!(t.contains("Coefficient") && t.contains("Model one"), "{t}");

    // collect copy (one name copies the current collection), style txt
    s.run("collect copy backup");
    s.run("collect set backup");
    s.run("collect style notes, noshownotes");
    s.run("collect style txt, delimiter(tab)");
    let e = path(&dir, "tab.txt");
    s.run(&format!("collect export \"{e}\""));
    let t = read(&e);
    assert!(t.contains("\tCoefficient\tStd. error"), "{t}");
    assert!(t.lines().any(|l| l.starts_with("x\t")), "{t}");
    assert!(!t.contains("Synthetic"), "{t}");
    s.run("collect set default");
    s.run("collect copy other, name(backup)");
    s.fails("collect copy other");
    s.run("collect copy other, replace");
    s.fails("collect style notes, bogus");
}

#[test]
fn putexcel_formulas_formats_pictures() {
    let dir = temp_dir("t51_xlsx");
    let x = path(&dir, "f.xlsx");
    let png = path(&dir, "i.png");
    // a 1x1 grey PNG
    let mut img = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR\0\0\0\x01\0\0\0\x01\x08\x00\0\0\0".to_vec();
    img.extend_from_slice(&[0x3a, 0x7e, 0x9b, 0x55]);
    std::fs::write(&png, img).unwrap();
    let mut s = Session::new();
    s.run(&format!("putexcel set \"{x}\", replace sheet(Out) open"));
    s.run("putexcel B1 = 2 B2 = 3");
    s.run("putexcel A1 = formula(SUM(B1:B2))");
    s.run("putexcel A2 = formula(=B1*B2+1), bold border(all) nformat(number_d2)");
    s.run("putexcel A3 = formula(VLOOKUP(1,B1:B2,1))");
    s.run("matrix M = (1,2\\3,4)");
    s.run("matrix rownames M = r1 r2");
    s.run("matrix colnames M = c1 c2");
    s.run("putexcel D1 = matrix(M), names hcenter");
    s.run("putexcel F1:G1 = \"merged\", merge hcenter vcenter txtwrap fpattern(solid, yellow) overwritefmt");
    s.run(&format!("putexcel H1 = image(\"{png}\")"));
    // nothing written before save in open mode
    assert!(!std::path::Path::new(&x).exists());
    s.run("putexcel save");
    let parts = ferrostat_cmd::xlsx::unzip(&std::fs::read(&x).unwrap()).unwrap();
    let sheet = String::from_utf8_lossy(&parts["xl/worksheets/sheet1.xml"]).to_string();
    assert!(sheet.contains("<f>SUM(B1:B2)</f><v>5</v>"), "{sheet}");
    assert!(sheet.contains("<f>B1*B2+1</f><v>7</v>"), "{sheet}");
    assert!(sheet.contains("<f>VLOOKUP(1,B1:B2,1)</f>"), "{sheet}");
    assert!(sheet.contains("<mergeCell ref=\"F1:G1\"/>"), "{sheet}");
    assert!(sheet.contains("<drawing r:id=\"rIdDr1\"/>"));
    assert!(parts.contains_key("xl/media/image1.png") && parts.contains_key("xl/drawings/drawing1.xml"));
    let ct = String::from_utf8_lossy(&parts["[Content_Types].xml"]).to_string();
    assert!(ct.contains("drawing+xml") && ct.contains("image/png"));
    let styles = String::from_utf8_lossy(&parts["xl/styles.xml"]).to_string();
    assert!(styles.contains("<b/>") && styles.contains("wrapText=\"1\""), "{styles}");
    // read back
    s.run(&format!("import excel using \"{x}\", sheet(Out) clear"));
    assert_eq!(s.nums("A")[..2], [5.0, 7.0]);
    assert_eq!(s.strs("E")[0], "c1");
    assert_eq!(s.strs("F")[0], "merged");
    // close discards unsaved changes
    s.run(&format!("putexcel set \"{x}\", modify sheet(Out) open"));
    s.run("putexcel B1 = 100");
    s.run("putexcel close");
    s.run(&format!("import excel using \"{x}\", sheet(Out) clear"));
    assert_eq!(s.nums("B")[0], 2.0);
    // without open every command writes the file; a value replaces a formula
    s.run(&format!("putexcel set \"{x}\", modify sheet(Out)"));
    s.run("putexcel A1 = 9");
    let parts = ferrostat_cmd::xlsx::unzip(&std::fs::read(&x).unwrap()).unwrap();
    let sheet = String::from_utf8_lossy(&parts["xl/worksheets/sheet1.xml"]).to_string();
    assert!(!sheet.contains("SUM(B1:B2)"), "{sheet}");
    s.fails(&format!("putexcel H2 = image(\"{}\")", path(&dir, "none.png")));
}

#[test]
fn docx_settings() {
    let dir = temp_dir("t51_set");
    let html = path(&dir, "a.html");
    std::fs::write(&html, "<html><body><p>first line\nsecond line</p></body></html>").unwrap();
    let mut s = Session::new();
    s.fails("set docx_hardbreak maybe");
    s.run(&format!("html2docx \"{html}\", saving(\"{}\")", path(&dir, "soft.docx")));
    s.run("set docx_hardbreak on");
    s.run(&format!("html2docx \"{html}\", saving(\"{}\")", path(&dir, "hard.docx")));
    s.run("set docx_paramode on");
    s.run(&format!("html2docx \"{html}\", saving(\"{}\")", path(&dir, "para.docx")));
    let doc = |f: &str| String::from_utf8_lossy(&ferrostat_cmd::xlsx::unzip(&std::fs::read(path(&dir, f)).unwrap()).unwrap()["word/document.xml"]).to_string();
    assert!(!doc("soft.docx").contains("<w:br/>"));
    assert!(doc("hard.docx").contains("<w:br/>"));
    let p = doc("para.docx");
    assert!(!p.contains("<w:br/>") && p.matches("<w:p>").count() >= 2, "{p}");
    s.run("set docx_hardbreak off");
    s.run("set docx_paramode off");
}

#[test]
fn ml_with_survey_design() {
    let mut s = Session::new();
    s.script(
        "set obs 200
         set seed 5
         gen strat = mod(_n,4)
         gen psu = ceil(_n/5)
         gen wt = 1 + mod(_n,3)
         gen x = rnormal()
         gen y = 1 + 0.5*x + rnormal()
         svyset psu [pw=wt], strata(strat)
         program myreg
           args lnf xb lnsig
           quietly replace `lnf' = ln(normalden($ML_y1, `xb', exp(`lnsig')))
         end",
    );
    s.run("svy: regress y x");
    let (b, se) = (s.scalar("_b[x]"), s.scalar("_se[x]"));
    s.run("ml model lf myreg (y = x) /lnsig, svy");
    s.run("ml maximize, nolog");
    assert_close(s.scalar("_b[eq1:x]"), b, 1e-5);
    assert_close(s.scalar("_se[eq1:x]"), se, 1e-5);
    assert_eq!(s.scalar("e(N_strata)"), 4.0);
    assert_eq!(s.scalar("e(N_psu)"), 160.0);
    assert_eq!(s.scalar("e(df_r)"), 156.0);
    assert_eq!(s.run("display e(vcetype)").trim(), "Linearized");

    s.run("svy, subpop(if x > 0): regress y x");
    let (b, se) = (s.scalar("_b[x]"), s.scalar("_se[x]"));
    s.run("ml model lf myreg (y = x) /lnsig, vce(linearized) subpop(if x > 0)");
    s.run("ml maximize, nolog");
    assert_close(s.scalar("_b[eq1:x]"), b, 1e-5);
    assert_close(s.scalar("_se[eq1:x]"), se, 1e-5);
    assert_eq!(s.scalar("e(N)"), 200.0);
    assert!(s.scalar("e(N_sub)") < 200.0);

    // replicate-weight designs are refused
    s.run("gen r1 = wt");
    s.run("svyset [pw=wt], brrweight(r1) vce(brr)");
    s.fails("ml model lf myreg (y = x) /lnsig, svy");
}
