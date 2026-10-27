//! Raw-text input (`infile` free format and with dictionaries, `infix`
//! fixed columns, inline dictionaries in do-files) and reading the early
//! binary .dta releases 102-112.

use std::path::Path;

use ferrostat_core::dta::DtaReader;
use ferrostat_core::VarType;
use ferrostat_tests::*;

fn write(dir: &Path, name: &str, text: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p.display().to_string()
}

fn is_nan_at(v: &[f64], i: usize) -> bool {
    v[i].is_nan()
}

const FIXED: &str = concat!(
    "   Bolivia            46       0        1\n",
    "   Brazil             74       0       10\n",
    "   Chile              89      16       29\n",
    "   TrinidadTobago     84      15       29\n",
);

#[test]
fn infile_free_format_with_strings() {
    let dir = temp_dir("t17_free");
    let raw = write(&dir, "f.raw", FIXED);
    let mut s = Session::new();
    let out = s.run(&format!("infile str14 country setting effort change using \"{raw}\""));
    assert!(out.contains("(4 observations read)"), "{out}");
    assert_eq!(s.varnames(), ["country", "setting", "effort", "change"]);
    assert_eq!(s.strs("country"), ["Bolivia", "Brazil", "Chile", "TrinidadTobago"]);
    assert_eq!(s.nums("setting"), [46., 74., 89., 84.]);
    assert_eq!(s.ds().get_var("country").unwrap().var_type, VarType::Str(14));
    assert_eq!(s.ds().get_var("setting").unwrap().var_type, VarType::Float);
    assert_close(s.scalar("change[4]"), 29.0, 0.0);
    // data in memory: clear is required
    s.fails(&format!("infile str14 country setting effort change using \"{raw}\""));
    // a string that is too long is truncated
    s.run(&format!("infile str5 country setting effort change using \"{raw}\", clear"));
    assert_eq!(s.strs("country")[3], "Trini");
    // if and in
    s.run(&format!("infile str14 country setting effort change using \"{raw}\" if setting > 50 in 2/4, clear"));
    assert_eq!(s.n(), 3);
    assert_eq!(s.nums("setting"), [74., 89., 84.]);
}

#[test]
fn infile_commas_quotes_skip_and_bad_numbers() {
    let dir = temp_dir("t17_commas");
    let raw = write(&dir, "a.raw", "1, \"New York\", 3.5, red\n2 'Los Angeles' . blue\n3,,4e1,red\n4 Chicago abc green\n");
    let mut s = Session::new();
    let out = s.run(&format!("infile id str12 city x str6 color using \"{raw}\""));
    assert!(out.contains("'abc' cannot be read as a number for x[4]"), "{out}");
    assert_eq!(s.strs("city"), ["New York", "Los Angeles", "", "Chicago"]);
    let x = s.nums("x");
    assert_eq!(x[0], 3.5);
    assert!(is_nan_at(&x, 1) && is_nan_at(&x, 3));
    assert_eq!(x[2], 40.0);
    assert_eq!(s.strs("color"), ["red", "blue", "red", "green"]);
    s.run(&format!("infile id _skip(2) str8 color using \"{raw}\", clear"));
    assert_eq!(s.varnames(), ["id", "color"]);
    assert_eq!(s.strs("color")[1], "blue");
    // automatic: text in numeric variables becomes value-labelled codes
    s.run(&format!("infile id _skip(2) color using \"{raw}\", clear automatic"));
    assert_eq!(s.nums("color"), [1., 2., 1., 3.]);
    let out = s.run("label list color");
    assert!(out.contains("3 green"), "{out}");
    assert_eq!(s.ds().get_var("color").unwrap().value_label_name, "color");
}

#[test]
fn infile_byvariable_ranges_and_missing_codes() {
    let dir = temp_dir("t17_byvar");
    let raw = write(&dir, "c.raw", "1 2 3 4\n10 20 30 .a\n");
    let mut s = Session::new();
    s.run(&format!("infile a b using \"{raw}\", byvariable(4)"));
    assert_eq!(s.n(), 4);
    assert_eq!(s.nums("a"), [1., 2., 3., 4.]);
    assert_eq!(s.scalar("b[4] == .a"), 1.0);
    s.run(&format!("infile v1-v4 using \"{raw}\", clear"));
    assert_eq!(s.varnames(), ["v1", "v2", "v3", "v4"]);
    assert_eq!(s.nums("v3"), [3., 30.]);
    s.run(&format!("infile byte v1 int v2 long v3 double v4 using \"{raw}\", clear"));
    assert_eq!(s.ds().get_var("v1").unwrap().var_type, VarType::Byte);
    assert_eq!(s.ds().get_var("v4").unwrap().var_type, VarType::Double);
}

#[test]
fn infile_regression_on_read_data() {
    // y = 0.6 + 0.8 x exactly in least squares for these values
    let dir = temp_dir("t17_reg");
    let raw = write(&dir, "r.raw", "1 1\n2 3\n3 2\n4 5\n5 4\n");
    let mut s = Session::new();
    s.run(&format!("infile x y using \"{raw}\""));
    s.run("regress y x");
    assert_close(s.scalar("_b[x]"), 0.8, 1e-12);
    assert_close(s.scalar("_b[_cons]"), 0.6, 1e-12);
}

#[test]
fn infix_columns_and_multiple_lines() {
    let dir = temp_dir("t17_infix");
    let raw = write(&dir, "f.raw", FIXED);
    let mut s = Session::new();
    let out = s.run(&format!("infix str country 4-17 setting 23-24 effort 31-32 change 40-41 using \"{raw}\""));
    assert!(out.contains("(4 observations read)"), "{out}");
    assert_eq!(s.ds().get_var("country").unwrap().var_type, VarType::Str(14));
    assert_eq!(s.strs("country")[3], "TrinidadTobago");
    assert_eq!(s.nums("effort"), [0., 0., 16., 15.]);
    assert_eq!(s.nums("change"), [1., 10., 29., 29.]);
    let two = write(&dir, "b.raw", "0001John      23\n  5.5  4.2\n0002Mary      31\n  6.1     \n");
    s.run(&format!("infix 2 lines 1: id 1-4 str name 5-14 int age 15-16 2: h 1-5 w 6-10 using \"{two}\", clear"));
    assert_eq!(s.n(), 2);
    assert_eq!(s.nums("id"), [1., 2.]);
    assert_eq!(s.strs("name"), ["John", "Mary"]);
    assert_eq!(s.ds().get_var("age").unwrap().var_type, VarType::Int);
    assert_close(s.nums("h")[1], 6.1, 1e-6);
    assert!(is_nan_at(&s.nums("w"), 1));
    // `/` moves to the next line
    s.run(&format!("infix id 1-4 str name 5-14 / h 1-5 w 2:6-10 using \"{two}\", clear"));
    assert_eq!(s.n(), 2);
    assert_close(s.nums("w")[0], 4.2, 1e-6);
}

#[test]
fn infix_dictionary_file_and_inline() {
    let dir = temp_dir("t17_infixdict");
    let raw = write(&dir, "effort.raw", FIXED);
    let dct = write(
        &dir,
        "effort.dct",
        &format!("infix dictionary using \"{raw}\" {{\n  str country  4-17\n\t  setting 23-24\n  /* comment */\n\t  effort  31-32\n\t  change  40-41\n}}\n"),
    );
    let mut s = Session::new();
    let out = s.run(&format!("infix using \"{dct}\""));
    assert!(out.contains("infix dictionary using"), "dictionary is displayed: {out}");
    assert!(out.contains("(4 observations read)"), "{out}");
    assert_eq!(s.nums("setting"), [46., 74., 89., 84.]);
    // the data file named in the dictionary can be replaced with using()
    let other = write(&dir, "o.raw", "   Peru               73       0        2\n");
    s.run(&format!("infix using \"{dct}\", using(\"{other}\") clear"));
    assert_eq!(s.n(), 1);
    assert_eq!(s.strs("country"), ["Peru"]);
    // inline in a do-file
    let dofile = write(
        &dir,
        "inline.do",
        &format!("infix dictionary using \"{raw}\", clear {{\n  str country  4-17\n\t  setting 23-24\n\t  effort  31-32\n\t  change  40-41\n}}\nsummarize change\n"),
    );
    let out = s.run(&format!("do \"{dofile}\""));
    assert!(out.contains("(4 observations read)"), "{out}");
    assert_eq!(s.n(), 4);
    assert_close(s.scalar("r(mean)"), 17.25, 1e-12);
}

#[test]
fn infile_dictionary_with_columns_labels_and_inline_data() {
    let dir = temp_dir("t17_infiledict");
    let dct = write(
        &dir,
        "d.dct",
        "* leading comment\ninfile dictionary {\n  _lines(2)\n  _line(1) id \"Identifier\"\n           str10 name %10s\n  _line(2) _column(3) inc %5.2f \"Income\"\n  /* c */ kids:kl\n}\n1 Ann Smith\n  12345 2\n2 Bob\n  99.5  3\n",
    );
    let mut s = Session::new();
    let out = s.run(&format!("infile using \"{dct}\""));
    assert!(out.contains("(2 observations read)"), "{out}");
    assert_eq!(s.strs("name"), ["Ann Smith", "Bob"]);
    // implied decimals with %5.2f when the field has no decimal point
    assert_close(s.nums("inc")[0], 123.45, 1e-4);
    assert_close(s.nums("inc")[1], 99.5, 1e-6);
    assert_eq!(s.nums("kids"), [2., 3.]);
    let ds = s.ds();
    assert_eq!(ds.get_var("id").unwrap().label, "Identifier");
    assert_eq!(ds.get_var("inc").unwrap().label, "Income");
    assert_eq!(ds.get_var("kids").unwrap().value_label_name, "kl");

    // separate data file with _firstlineoffile and _column/_skip
    let raw = write(&dir, "x.raw", "header line\nAAA 12 x 7\nBBB 34 y 8\n");
    let d2 = write(&dir, "d2.dct", &format!("dictionary using \"{raw}\" {{\n _firstlineoffile(2)\n str3 code %3s\n _column(5) int a %2f\n _skip(3) b\n}}\n"));
    s.run(&format!("infile using \"{d2}\", clear"));
    assert_eq!(s.strs("code"), ["AAA", "BBB"]);
    assert_eq!(s.nums("a"), [12., 34.]);
    assert_eq!(s.nums("b"), [7., 8.]);
}

// ---- early .dta releases -------------------------------------------------

fn fixed(buf: &mut Vec<u8>, s: &str, w: usize) {
    let mut b = s.as_bytes().to_vec();
    b.resize(w, 0);
    buf.extend_from_slice(&b);
}

/// A file in release `rel` (104, 105, 108, 110, 111) with variables
/// id (byte), name (str4), pop (int), inc (long), w (float), d (double).
fn old_dta(rel: u8) -> Vec<u8> {
    let char_types = rel <= 110;
    let name_w = if rel >= 110 { 33 } else { 9 };
    let fmt_w = if rel >= 105 { 12 } else { 7 };
    let lbl_w = if rel > 105 { 81 } else { 32 };
    let mut b = vec![rel, 2, 1, 0];
    b.extend_from_slice(&6u16.to_le_bytes());
    b.extend_from_slice(&3u32.to_le_bytes());
    fixed(&mut b, "old data", lbl_w);
    if rel > 104 {
        fixed(&mut b, "24 Feb 2001 14:23", 18);
    }
    if char_types {
        b.extend_from_slice(&[b'b', 0x7f + 4, b'i', b'l', b'f', b'd']);
    } else {
        b.extend_from_slice(&[251, 4, 252, 253, 254, 255]);
    }
    for n in ["id", "name", "pop", "inc", "w", "d"] {
        fixed(&mut b, n, name_w);
    }
    // sorted by id; garbage after the terminator must be ignored
    for v in [1u16, 0, 5, 3, 0, 0, 0] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for f in ["%8.0g", "%4s", "%8.0g", "%12.0g", "%9.0g", "%10.0g"] {
        fixed(&mut b, f, fmt_w);
    }
    for l in ["idl", "", "", "", "", ""] {
        fixed(&mut b, l, name_w);
    }
    for l in ["identifier", "", "population", "", "", "double"] {
        fixed(&mut b, l, lbl_w);
    }
    if rel > 104 {
        b.push(0);
        if rel <= 108 {
            b.extend_from_slice(&0u16.to_le_bytes());
        } else {
            b.extend_from_slice(&0u32.to_le_bytes());
        }
    }
    // observations: the old missing codes are the type maxima, and byte
    // values above 100 were ordinary numbers
    let rows: [(i8, &str, i16, i32, f32, f64); 3] = [
        (1, "ab", 32767, 100000, 1.5, 0.25),
        (110, "cdef", -5, i32::MAX, f32::from_bits(0x7f00_0000), -1.0),
        (i8::MAX, "", 32000, -7, 2.0, f64::from_bits(0x7fe0_0000_0000_0000)),
    ];
    for (id, name, pop, inc, w, d) in rows {
        b.push(id as u8);
        fixed(&mut b, name, 4);
        b.extend_from_slice(&pop.to_le_bytes());
        b.extend_from_slice(&inc.to_le_bytes());
        b.extend_from_slice(&w.to_le_bytes());
        b.extend_from_slice(&d.to_le_bytes());
    }
    // value label idl: 1 "one", 110 "big"
    if rel < 106 {
        b.extend_from_slice(&20u16.to_le_bytes());
        fixed(&mut b, "idl", 9);
        b.push(0);
        b.extend_from_slice(&1i16.to_le_bytes());
        b.extend_from_slice(&110i16.to_le_bytes());
        fixed(&mut b, "one", 8);
        fixed(&mut b, "big", 8);
    } else {
        let txt = b"one\0big\0";
        let mut t = Vec::new();
        t.extend_from_slice(&2u32.to_le_bytes());
        t.extend_from_slice(&(txt.len() as u32).to_le_bytes());
        t.extend_from_slice(&0u32.to_le_bytes());
        t.extend_from_slice(&4u32.to_le_bytes());
        t.extend_from_slice(&1i32.to_le_bytes());
        t.extend_from_slice(&110i32.to_le_bytes());
        t.extend_from_slice(txt);
        b.extend_from_slice(&(t.len() as u32).to_le_bytes());
        fixed(&mut b, "idl", name_w);
        b.extend_from_slice(&[0, 0, 0]);
        b.extend_from_slice(&t);
    }
    b
}

#[test]
fn old_dta_releases_are_readable() {
    for rel in [104u8, 105, 108, 110, 111] {
        let ds = DtaReader::from_bytes(old_dta(rel)).unwrap_or_else(|e| panic!("release {rel}: {e}"));
        assert_eq!(ds.n_obs, 3, "release {rel}");
        assert_eq!(ds.varlist(), ["id", "name", "pop", "inc", "w", "d"], "release {rel}");
        assert_eq!(ds.datalabel, "old data");
        assert_eq!(ds.sort_by, ["id"], "release {rel}");
        let num = |v: &str| ds.get_var(v).unwrap().data.iter().map(|x| x.num()).collect::<Vec<f64>>();
        // byte column holds 110, so it is widened
        assert_eq!(ds.get_var("id").unwrap().var_type, VarType::Int, "release {rel}");
        let id = num("id");
        assert_eq!(&id[..2], [1.0, 110.0]);
        assert!(ferrostat_core::is_missing(id[2]));
        assert_eq!(ds.get_var("name").unwrap().var_type, VarType::Str(4));
        assert_eq!(ds.get_var("name").unwrap().str_at(1), "cdef");
        let pop = num("pop");
        assert!(ferrostat_core::is_missing(pop[0]));
        assert_eq!(&pop[1..], [-5.0, 32000.0]);
        assert_eq!(ds.get_var("pop").unwrap().var_type, VarType::Int);
        let inc = num("inc");
        assert_eq!(inc[0], 100000.0);
        assert!(ferrostat_core::is_missing(inc[1]));
        assert!(ferrostat_core::is_missing(num("w")[1]));
        assert!(ferrostat_core::is_missing(num("d")[2]));
        assert_eq!(ds.get_var("pop").unwrap().label, "population");
        assert_eq!(ds.get_var("id").unwrap().value_label_name, "idl");
        let vl = ds.value_labels.get("idl").unwrap_or_else(|| panic!("release {rel}: no value label"));
        assert_eq!(vl.lookup(110.0), Some("big"));
        assert_eq!(vl.lookup(1.0), Some("one"));
    }
}

#[test]
fn use_old_release_and_analyse() {
    let dir = temp_dir("t17_olduse");
    let p = dir.join("old108.dta");
    std::fs::write(&p, old_dta(108)).unwrap();
    let mut s = Session::new();
    s.run(&format!("use \"{}\"", p.display()));
    assert_eq!(s.n(), 3);
    s.run("summarize pop");
    assert_eq!(s.scalar("r(N)"), 2.0);
    assert_close(s.scalar("r(mean)"), 15997.5, 1e-9);
    // re-saving in the current format keeps the widened values
    let q = dir.join("new.dta");
    s.run(&format!("save \"{}\"", q.display()));
    s.run(&format!("use \"{}\", clear", q.display()));
    assert_eq!(s.scalar("id[2]"), 110.0);
}
