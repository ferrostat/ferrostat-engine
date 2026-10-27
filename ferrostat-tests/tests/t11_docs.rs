//! Dynamic documents: putdocx, putpdf, dyndoc, dyntext, markdown.

use ferrostat_tests::*;

fn document_xml(path: &std::path::Path) -> String {
    let bytes = std::fs::read(path).expect("docx written");
    let parts = ferrostat_cmd::xlsx::unzip(&bytes).expect("valid zip");
    for p in ["[Content_Types].xml", "_rels/.rels", "word/styles.xml", "word/_rels/document.xml.rels"] {
        assert!(parts.contains_key(p), "missing part {p}");
    }
    String::from_utf8(parts["word/document.xml"].clone()).unwrap()
}

#[test]
fn putdocx_paragraphs_text_and_tables() {
    let dir = temp_dir("putdocx");
    let file = dir.join("report.docx");
    let mut s = Session::with_fixture();
    s.script(&format!(
        r#"
putdocx begin, pagesize(A4)
putdocx paragraph, style(Title)
putdocx text ("Report & summary")
quietly summarize y
putdocx paragraph, halign(center)
putdocx text ("Mean: ")
putdocx text (r(mean)), nformat(%5.2f) bold
putdocx text (" red"), font(,,red) italic
putdocx pagebreak
putdocx table t1 = data(x y) in 1/3, varnames
putdocx table t1(1,.), bold
putdocx table t2 = (2,2)
putdocx table t2(1,2) = ("cell")
regress y x
putdocx table t3 = etable
putdocx save "{}", replace
"#,
        file.display()
    ));
    let x = document_xml(&file);
    assert!(x.contains(r#"<w:pStyle w:val="Title"/>"#));
    assert!(x.contains("Report &amp; summary"));
    assert!(x.contains(">13.01<"), "formatted mean");
    assert!(x.contains(r#"<w:jc w:val="center"/>"#));
    assert!(x.contains(r#"<w:color w:val="FF0000"/>"#));
    assert!(x.contains(r#"<w:br w:type="page"/>"#));
    assert!(x.contains(">cell<"));
    assert!(x.contains(">_cons<"));
    assert_eq!(x.matches("<w:tbl>").count(), 3);
    // the document is closed by save
    s.fails("putdocx text (\"x\")");
    // replace is required to overwrite
    s.run("putdocx begin");
    s.fails(&format!("putdocx save \"{}\"", file.display()));
    s.run("putdocx clear");
}

#[test]
fn putdocx_image_and_append() {
    let dir = temp_dir("putdocx_img");
    let png = dir.join("g.png");
    let file = dir.join("img.docx");
    let mut s = Session::with_fixture();
    s.script(&format!(
        r#"
scatter y x
graph export "{png}", replace
putdocx begin
putdocx paragraph
putdocx image "{png}", width(10 cm)
putdocx save "{file}", replace
putdocx begin
putdocx paragraph, style(Heading1)
putdocx text ("More")
putdocx save "{file}", append
"#,
        png = png.display(),
        file = file.display()
    ));
    let x = document_xml(&file);
    assert!(x.contains(r#"cx="3600000""#), "10 cm wide");
    assert!(x.contains("<wp:inline"));
    assert!(x.contains(">More<"));
    let parts = ferrostat_cmd::xlsx::unzip(&std::fs::read(&file).unwrap()).unwrap();
    assert!(parts.keys().any(|k| k.starts_with("word/media/")));
}

#[test]
fn putpdf_writes_pdf() {
    let dir = temp_dir("putpdf");
    let file = dir.join("out.pdf");
    let mut s = Session::with_fixture();
    s.script(&format!(
        r#"
putpdf begin
putpdf paragraph, style(Heading1)
putpdf text ("Hello (pdf)")
putpdf table t = (1,2)
putpdf table t(1,1) = ("a")
putpdf save "{}", replace
"#,
        file.display()
    ));
    let b = std::fs::read(&file).unwrap();
    assert!(b.starts_with(b"%PDF-1.4"));
    let t = String::from_utf8_lossy(&b);
    assert!(t.contains(r"(\(pdf\)) Tj"));
    assert!(t.contains("%%EOF"));
}

#[test]
fn dyndoc_runs_code_and_converts_markdown() {
    let dir = temp_dir("dyndoc");
    let src = dir.join("doc.txt");
    std::fs::write(
        &src,
        "<<dd_version: 2>>\nTitle\n=====\n\n<<dd_do: quietly>>\nsummarize y\nlocal m = round(r(mean), .01)\n<</dd_do>>\n\n## Mean is <<dd_display: `m'>>\n\n<<dd_do>>\ndisplay 6*7\n<</dd_do>>\n\n<<dd_skip_if: `m' > 0>>\nskipped\n<<dd_skip_else>>\nkept **bold**\n<<dd_skip_end>>\n\n<<dd_ignore>>\n<<dd_display: 1>>\n<</dd_ignore>>\n<<dd_remove>>gone<</dd_remove>>\n",
    )
    .unwrap();
    let mut s = Session::with_fixture();
    let html = dir.join("doc.html");
    s.run(&format!("dyndoc \"{}\", replace", src.display()));
    let h = std::fs::read_to_string(&html).unwrap();
    assert!(h.contains("<h1>Title</h1>"));
    assert!(h.contains("<h2>Mean is 13.01</h2>"));
    assert!(h.contains(". display 6*7\n42"));
    assert!(!h.contains("summarize y"), "quietly hides commands");
    assert!(h.contains("kept <strong>bold</strong>"));
    assert!(!h.contains("skipped"));
    assert!(h.contains("&lt;&lt;dd_display: 1&gt;&gt;"));
    assert!(!h.contains("gone"));
    // the document's locals do not leak into the session
    assert_eq!(s.run("display \"`m'\""), "");
    // existing output requires replace
    s.fails(&format!("dyndoc \"{}\"", src.display()));

    s.run(&format!("dyndoc \"{}\", docx replace", src.display()));
    let x = document_xml(&dir.join("doc.docx"));
    assert!(x.contains(r#"<w:pStyle w:val="Heading2"/>"#));
    assert!(x.contains("Mean is 13.01"));
    assert!(x.contains("Courier New"));

    let out = dir.join("doc.out");
    s.run(&format!("dyntext \"{}\", saving(\"{}\") replace", src.display(), out.display()));
    let t = std::fs::read_to_string(&out).unwrap();
    assert!(t.starts_with("Title\n=====\n"));
    assert!(t.contains(". display 6*7\n42\n"));
    assert!(!t.contains("~~~~"));
}

#[test]
fn dyndoc_graph_and_markdown_command() {
    let dir = temp_dir("dyndoc_graph");
    let src = dir.join("g.txt");
    let png = dir.join("plot.png");
    std::fs::write(&src, format!("<h1>Plot</h1>\n\n<<dd_do: quietly>>\nscatter y x\n<</dd_do>>\n\n<<dd_graph: saving(\"{}\") replace height(400)>>\n", png.display())).unwrap();
    let mut s = Session::with_fixture();
    s.run(&format!("dyndoc \"{}\", docx replace", src.display()));
    let x = document_xml(&dir.join("g.docx"));
    assert!(x.contains(r#"<w:pStyle w:val="Heading1"/>"#));
    assert!(x.contains("<wp:inline"));
    assert!(png.exists());

    let md = dir.join("m.md");
    std::fs::write(&md, "# A\n\n- one\n- two\n").unwrap();
    let html = dir.join("m.html");
    s.run(&format!("markdown \"{}\", saving(\"{}\") replace", md.display(), html.display()));
    let h = std::fs::read_to_string(&html).unwrap();
    assert!(h.contains("<li>two</li>"));
}
