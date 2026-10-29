//! Remaining data-management commands: icd9/icd9p/icd10/icd10cm/icd10pcs,
//! vl, bcal and %tb formats, memory, snapshot, sysuse, frame sets on disk
//! (frames save/use/describe/modify), frame put label, fralias/frunalias,
//! import fred (local files), export/import dbase, SAS transport round
//! trips, fixed-format infile/infix dictionaries, unicode helpers, dyngen,
//! putmata/getmata.

use ferrostat_tests::*;

fn path(dir: &std::path::Path, f: &str) -> String {
    dir.join(f).to_string_lossy().replace('\\', "/")
}

#[test]
fn icd_structure_check_clean_generate() {
    let mut s = Session::new();
    s.script(
        r#"set obs 7
           gen str10 code = ""
           replace code = "250.01" in 1
           replace code = "E812.0" in 2
           replace code = "V10.3" in 3
           replace code = "25.001" in 4
           replace code = "abc" in 5
           replace code = "4011" in 7"#,
    );
    let out = s.run("icd9 check code, generate(bad)");
    assert!(out.contains("invalid codes"), "{out}");
    assert_eq!(s.scalar("r(esum)"), 2.0);
    assert_eq!(s.scalar("r(e1)"), 1.0);
    assert_eq!(s.scalar("r(e5)"), 1.0);
    assert_eq!(s.nums("bad"), vec![0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0]);
    s.run("icd9 generate cat = code, category");
    assert_eq!(s.strs("cat"), vec!["250", "E812", "V10", "", "", "", "401"]);
    s.run("icd9 generate inr = code, range(250/251 V10*)");
    let inr = s.nums("inr");
    assert_eq!(&inr[..3], &[1.0, 0.0, 1.0]);
    assert_eq!(inr[6], 0.0);
    // clean refuses invalid codes, then works once they are gone
    s.fails("icd9 clean code");
    s.run("drop if bad");
    s.run("icd9 clean code");
    assert_eq!(s.strs("code"), vec!["250.01", "E812.0", "V10.3", "", "401.1"]);
    s.run("icd9 clean code, nodots");
    assert_eq!(s.strs("code")[0], "25001");

    let mut s = Session::new();
    s.script(
        r#"set obs 3
           gen str10 c = "a27.0"
           replace c = "S52521A" in 2
           replace c = "G40.9" in 3"#,
    );
    s.run("icd10cm check c");
    assert_eq!(s.scalar("r(esum)"), 0.0);
    s.fails("icd10cm clean c");
    s.run("icd10cm clean c, generate(c2)");
    assert_eq!(s.strs("c2"), vec!["A27.0", "S52.521A", "G40.9"]);
    s.run("icd10cm generate cat = c, category");
    assert_eq!(s.strs("cat"), vec!["A27", "S52", "G40"]);
    let out = s.run("icd10cm generate d = c, description");
    assert!(out.contains("not distributed"), "{out}");
    assert_eq!(s.strs("d"), vec!["", "", ""]);
    // the international edition has no letters after the category digits
    s.run("icd10 check c, generate(e10)");
    assert_eq!(s.nums("e10"), vec![0.0, 4.0, 0.0]);
    s.run("icd10 generate sh = c2, short");
    assert_eq!(s.strs("sh")[0], "A27.0");
    // procedure codes: seven characters, no I or O, no period
    s.run(r#"replace c = "0DTJ4ZZ" in 1"#);
    s.run(r#"replace c = "0DTI4ZZ" in 2"#);
    s.run("icd10pcs check c, generate(ep)");
    assert_eq!(s.nums("ep"), vec![0.0, 5.0, 1.0]);
    s.run("icd9p check c");
    assert_eq!(s.scalar("r(esum)"), 3.0);
    assert!(s.run("icd10 lookup A27.0 G40*").contains("not available"));
    s.run("icd10 query");
}

#[test]
fn vl_lists_and_rebuild() {
    let dir = temp_dir("t48_vl");
    let mut s = Session::new();
    s.script(
        "set obs 200
         gen id = _n
         gen d = mod(_n, 2)
         gen c3 = mod(_n, 3)
         gen u = mod(_n, 40)
         gen x = _n / 7
         gen k = 5",
    );
    s.run("vl set, dummy");
    assert_eq!(s.run("display \"$vldummy\"").trim(), "d");
    assert_eq!(s.run("display \"$vlcategorical\"").trim(), "c3");
    assert_eq!(s.run("display \"$vluncertain\"").trim(), "u");
    assert_eq!(s.run("display \"$vlcontinuous\"").trim(), "id x");
    assert_eq!(s.run("display \"$vlother\"").trim(), "k");
    assert_eq!(s.scalar("r(k_continuous)"), 2.0);
    s.run("vl create mine = vlcontinuous + (d) - (id)");
    assert_eq!(s.run("display \"$mine\"").trim(), "x d");
    s.fails("vl create mine = (x)");
    s.run("vl modify mine = mine + vlother");
    assert_eq!(s.run("display \"$mine\"").trim(), "x d k");
    s.run("vl label mine \"my variables\"");
    let out = s.run("vl dir");
    assert!(out.contains("$mine") && out.contains("my variables"), "{out}");
    s.run("vl move (u) vlcontinuous");
    assert_eq!(s.run("display \"$vlcontinuous\"").trim(), "id u x");
    s.run("vl substitute f = i.vldummy c.mine");
    assert_eq!(s.run("display \"$f\"").trim(), "i.d c.x c.d c.k");
    s.run("vl substitute g = vldummy#vlcategorical");
    assert_eq!(s.run("display \"$g\"").trim(), "d#c3");
    // lists live with the data: save, forget the macros, rebuild
    let p = path(&dir, "vl.dta");
    s.run("drop k");
    s.run(&format!("save \"{p}\", replace"));
    s.run("macro drop mine");
    s.run(&format!("use \"{p}\", clear"));
    s.run("vl rebuild");
    assert_eq!(s.run("display \"$mine\"").trim(), "x d");
    s.run("vl drop (d)");
    assert_eq!(s.run("display \"$mine\"").trim(), "x");
    s.run("vl drop mine");
    assert!(!s.run("vl dir").contains("$mine"));
    s.run("vl clear");
    assert!(s.run("vl dir").contains("no variable lists"));
}

#[test]
fn bcal_calendar_files_and_tb_format() {
    let dir = temp_dir("t48_bcal");
    let mut s = Session::new();
    s.run(&format!("cd \"{}\"", dir.to_string_lossy()));
    std::fs::write(
        dir.join("wk.stbcal"),
        "* weekdays of 2020\nversion 17\npurpose \"Weekdays\"\ndateformat ymd\nrange 2020-01-01 2020-12-31\ncenterdate 2020-01-02\nomit dayofweek (Sa Su)\nomit date 2020-01-01\nomit date *-12-25 and +1 if dow(Fr)\nomit dowinmonth +3 mo of jan\n",
    )
    .unwrap();
    let out = s.run("bcal describe wk");
    assert!(out.contains("Weekdays"), "{out}");
    // 262 weekdays in 2020, minus 1 Jan, 25 Dec and the third Monday of January
    assert_eq!(s.scalar("r(included)"), 259.0);
    assert_eq!(s.scalar("r(omitted)"), 366.0 - 259.0);
    assert!(s.run("bcal dir").contains("wk"));
    s.run("bcal check wk");
    s.script(
        "set obs 21
         gen td = mdy(1,1,2020) + _n
         gen b = bofd(\"wk\", td)
         gen back = dofb(b, \"wk\")
         format b %tbwk",
    );
    let b = s.nums("b");
    // 2 Jan is business day 0, 3 Jan is 1, 4-5 Jan weekend, 6 Jan is 2
    assert_eq!((b[0], b[1], b[4]), (0.0, 1.0, 2.0));
    assert!(b[2].is_nan() && b[3].is_nan());
    // 20 Jan 2020 (third Monday) is omitted
    assert!(b[18].is_nan());
    let td = s.nums("td");
    let back = s.nums("back");
    for i in 0..21 {
        if !b[i].is_nan() {
            assert_eq!(back[i], td[i]);
        }
    }
    let out = s.run("list b in 5");
    assert!(out.contains("06jan2020"), "{out}");
    assert_eq!(s.run("display %tbwk 2").trim(), "06jan2020");
    // create a calendar from the dates present in the data
    s.run("keep if !missing(b)");
    s.run("bcal create mine, from(td) generate(bm) purpose(\"from data\") replace");
    assert_eq!(s.nums("bm"), s.nums("b"));
    s.run("bcal describe mine");
    assert_eq!(s.scalar("r(included)"), s.n() as f64);
    let text = std::fs::read_to_string(dir.join("mine.stbcal")).unwrap();
    assert!(text.contains("omit dayofweek (Su Sa)") || text.contains("omit dayofweek"), "{text}");
    s.fails("bcal describe nosuchcalendar");
}

#[test]
fn memory_snapshot_sysuse() {
    let mut s = Session::new();
    s.run("set obs 100");
    s.run("gen double x = _n");
    s.run("gen byte y = 1");
    let out = s.run("memory");
    assert!(out.contains("Memory usage"), "{out}");
    assert_eq!(s.scalar("r(data_u)"), 900.0);
    s.run("set max_memory 16g");
    s.run("set segmentsize 64m");
    assert!(s.run("memory").contains("64m"));

    s.run("snapshot save, label(\"full\")");
    assert_eq!(s.scalar("r(snapshot)"), 1.0);
    s.run("drop if x > 10");
    s.run("snapshot save");
    s.run("snapshot label 2 \"ten\"");
    let out = s.run("snapshot list");
    assert!(out.contains("full") && out.contains("ten"), "{out}");
    s.run("snapshot restore 1");
    assert_eq!(s.n(), 100);
    s.run("snapshot restore 2");
    assert_eq!(s.n(), 10);
    s.run("snapshot erase 1");
    s.fails("snapshot restore 1");
    s.run("snapshot erase _all");
    assert!(s.run("snapshot list").contains("no snapshots"));

    let out = s.run("sysuse dir");
    assert!(out.contains("wagepanel.dta") && out.contains("trial.dta"), "{out}");
    s.fails("sysuse wagepanel");
    s.run("sysuse wagepanel, clear");
    assert_eq!(s.n(), 1200);
    s.run("xtset id year");
    s.run("xtreg lwage educ exper female union, re");
    let b = s.scalar("_b[educ]");
    assert!((b - 0.08).abs() < 0.03, "{b}");
    s.run("sysuse households, clear");
    assert_eq!(s.n(), 500);
    s.run("sysuse salesmonthly, clear");
    s.run("tsset month");
    s.run("sysuse trial, clear");
    s.run("stset time, failure(died)");
    s.fails("sysuse nosuchdata, clear");
}

#[test]
fn frame_sets_and_aliases() {
    let dir = temp_dir("t48_frames");
    let fs = path(&dir, "set");
    let mut s = Session::new();
    s.script(
        "set obs 4
         gen id = _n
         gen x = 10*_n
         label define lab 1 one 2 two
         label values id lab
         frame create other
         frame other: set obs 2
         frame other: gen k = _n",
    );
    s.run("frame put label lab, into(other)");
    let out = s.run("frame other: label list");
    assert!(out.contains("one"), "{out}");
    s.run(&format!("frames save \"{fs}\", frames(_all) replace"));
    assert_eq!(s.run("display \"`r(frames)'\"").trim(), "default other");
    let out = s.run(&format!("frames describe using \"{fs}\""));
    assert!(out.contains("Frame other") && out.contains("Observations:          2"), "{out}");
    s.run("frame create extra");
    s.run("frame extra: set obs 3");
    s.run("frame extra: gen z = 1");
    s.run(&format!("frames modify \"{fs}\", add(extra)"));
    assert_eq!(s.run("display \"`r(frames)'\"").trim(), "default other extra");
    s.fails(&format!("frames modify \"{fs}\", add(extra)"));
    s.run(&format!("frames modify \"{fs}\", add(extra) replace"));
    s.run(&format!("frames modify \"{fs}\", drop(other)"));
    assert_eq!(s.run("display \"`r(frames)'\"").trim(), "default extra");
    s.fails(&format!("frames use \"{fs}\""));
    s.run(&format!("frames use \"{fs}\", clear"));
    assert_eq!(s.run("display \"`r(frames)'\"").trim(), "default extra");
    assert_eq!(s.n(), 4);
    assert_eq!(s.run("frame extra: count").trim(), "3");
    s.fails("frame other: count");
    // replace: reload one frame over the one in memory
    s.run("frame extra: drop in 1");
    s.run(&format!("frames use \"{fs}\", frames(extra) replace"));
    assert_eq!(s.run("frame extra: count").trim(), "3");
    let out = s.run("frames describe extra, short");
    assert!(out.contains("Frame extra"), "{out}");

    // alias variables
    s.script(
        "frame create look
         frame look: set obs 2
         frame look: gen k = _n
         frame look: gen w = 7*_n
         gen k = mod(_n, 2) + 1",
    );
    s.run("frlink m:1 k, frame(look)");
    s.run("fralias add w, from(look)");
    s.run("fralias add w2 = w, from(look)");
    assert_eq!(s.nums("w"), vec![14.0, 7.0, 14.0, 7.0]);
    assert_eq!(s.nums("w2"), s.nums("w"));
    let out = s.run("fralias describe");
    assert!(out.contains("w2") && out.contains("look"), "{out}");
    s.run("frunalias w");
    let out = s.run("fralias describe");
    assert!(!out.contains("  w   ") && out.contains("w2"), "{out}");
    s.run("frunalias");
    assert!(s.run("fralias describe").contains("no alias"));
}

#[test]
fn fred_files_and_dbase_round_trip() {
    let dir = temp_dir("t48_fred");
    std::fs::write(dir.join("GDPX.json"), r#"{"observations":[{"date":"2020-01-01","value":"100.5"},{"date":"2020-04-01","value":"."},{"date":"2020-07-01","value":"102.25"}]}"#).unwrap();
    std::fs::write(dir.join("RATEX.csv"), "DATE,RATEX\n2020-01-01,3.5\n2020-04-01,14.7\n2020-10-01,6.9\n").unwrap();
    let d = dir.to_string_lossy().replace('\\', "/");
    let mut s = Session::new();
    s.run(&format!("import fred GDPX RATEX, cachedir(\"{d}\") clear"));
    assert_eq!(s.n(), 4);
    assert_eq!(s.strs("datestr"), vec!["2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01"]);
    let g = s.nums("GDPX");
    assert_eq!(g[0], 100.5);
    assert!(g[1].is_nan() && g[3].is_nan());
    assert_eq!(s.nums("RATEX")[1], 14.7);
    s.run(&format!("import fred GDPX, cachedir(\"{d}\") daterange(2020-02-01 .) clear"));
    assert_eq!(s.n(), 2);
    let err = s.try_run("import fred NOPE, clear").unwrap_err();
    assert!(err.contains("no network access"), "{err}");

    // dBase: export and read back
    let p = path(&dir, "t.dbf");
    let mut s = Session::new();
    s.script(
        "set obs 5
         gen id = _n
         gen double v = _n / 4
         gen str6 name = \"n\" + string(_n)
         replace v = . in 3",
    );
    s.run(&format!("export dbase using \"{p}\", replace"));
    s.fails(&format!("export dbase using \"{p}\""));
    s.run(&format!("import dbase using \"{p}\", clear"));
    assert_eq!(s.nums("id"), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let v = s.nums("v");
    assert_eq!(v[0], 0.25);
    assert!(v[2].is_nan());
    assert_eq!(s.strs("name")[4], "n5");
    s.fails("odbc list");
    s.fails("jdbc connect");
    s.fails("import haver x@usecon");
}

#[test]
fn sas_transport_round_trip() {
    let dir = temp_dir("t48_xpt");
    for (v, f) in [("5", "a.xpt"), ("8", "b.xpt")] {
        let p = path(&dir, f);
        let mut s = Session::new();
        s.script(
            "set obs 4
             gen double a = _n * 1.5
             gen str4 s = \"s\" + string(_n)
             replace a = . in 2",
        );
        s.run(&format!("export sasxport{v} \"{p}\", replace"));
        s.run(&format!("import sasxport{v} \"{p}\", clear"));
        let a = s.nums("a");
        assert_eq!(a[0], 1.5);
        assert!(a[1].is_nan());
        assert_eq!(s.strs("s")[3], "s4");
    }
}

#[test]
fn fixed_format_dictionaries() {
    let dir = temp_dir("t48_dct");
    std::fs::write(dir.join("d.raw"), "  1 alpha\n  12.500\n  2 beta \n   3.250\n").unwrap();
    std::fs::write(
        dir.join("d.dct"),
        format!(
            "dictionary using \"{}\" {{\n  _lines(2)\n  _line(1)\n  _column(1) int id %3f\n  _skip(1)\n  str5 name %5s\n  _line(2)\n  _column(2) double val %6f \"value label\"\n}}\n",
            path(&dir, "d.raw")
        ),
    )
    .unwrap();
    std::fs::write(
        dir.join("i.dct"),
        format!("infix dictionary using \"{}\" {{\n  2 lines\n  1:\n  id 1-3\n  str name 5-9\n  2:\n  val 2-7\n}}\n", path(&dir, "d.raw")),
    )
    .unwrap();
    std::fs::write(dir.join("n.dct"), "dictionary {\n  int a\n  _newline\n  float b\n}\n1\n2.5\n3\n4.5\n").unwrap();
    let mut s = Session::new();
    s.run(&format!("infile using \"{}\", clear", path(&dir, "d.dct")));
    assert_eq!(s.nums("id"), vec![1.0, 2.0]);
    assert_eq!(s.strs("name"), vec!["alpha", "beta"]);
    assert_eq!(s.nums("val"), vec![12.5, 3.25]);
    s.run(&format!("infix using \"{}\", clear", path(&dir, "i.dct")));
    assert_eq!(s.nums("val"), vec![12.5, 3.25]);
    assert_eq!(s.strs("name"), vec!["alpha", "beta"]);
    s.run(&format!("infile using \"{}\", clear", path(&dir, "n.dct")));
    assert_eq!(s.nums("a"), vec![1.0, 3.0]);
    assert_eq!(s.nums("b"), vec![2.5, 4.5]);
}

#[test]
fn unicode_helpers() {
    let dir = temp_dir("t48_uni");
    let f = path(&dir, "l1.txt");
    std::fs::write(&f, b"caf\xe9\n").unwrap();
    let mut s = Session::new();
    s.run(&format!("unicode analyze \"{f}\""));
    assert_eq!(s.scalar("r(N_needs_translation)"), 1.0);
    s.run(&format!("unicode translate \"{f}\", encoding(latin1)"));
    assert_eq!(std::fs::read_to_string(&f).unwrap(), "café\n");
    s.run(&format!("unicode retranslate \"{f}\", encoding(cp1252)"));
    assert_eq!(std::fs::read_to_string(&f).unwrap(), "café\n");
    s.run(&format!("unicode restore \"{f}\""));
    assert_eq!(std::fs::read(&f).unwrap(), b"caf\xe9\n");
    let g = path(&dir, "u8.txt");
    s.run(&format!("unicode convertfile \"{f}\" \"{g}\", srcencoding(latin1) dstencoding(utf8) replace"));
    assert_eq!(std::fs::read_to_string(&g).unwrap(), "café\n");
    assert!(s.run("unicode encoding list").contains("latin1"));
    assert!(s.run("unicode encoding alias l1").contains("iso-8859-1"));
    s.run("unicode locale list");
    s.run("unicode collator list");
}

#[test]
fn dyngen_simultaneous_updates() {
    let mut s = Session::new();
    s.script(
        "set obs 5
         gen t = _n
         gen x1 = t + 2
         gen x2 = mod(t*3, 5)
         gen double d1 = x1
         gen double d2 = x2",
    );
    s.run("dyngen {\n update d1 = .4*d1 + .1*d2[_n-1], missval(10)\n update d2 = .2*d1[_n-1] + .3*d2[_n-1], missval(20)\n}");
    // closed form, observation by observation
    let (x1, mut d1, mut d2) = (s.nums("x1"), vec![0.0; 5], vec![0.0; 5]);
    for i in 0..5 {
        let d2l = if i == 0 { 10.0 } else { d2[i - 1] };
        d1[i] = 0.4 * x1[i] + 0.1 * d2l;
        let (d1l, d2l2) = if i == 0 { (20.0, 20.0) } else { (d1[i - 1], d2[i - 1]) };
        d2[i] = 0.2 * d1l + 0.3 * d2l2;
    }
    assert_all_close(&s.nums("d1"), &d1, 1e-12);
    assert_all_close(&s.nums("d2"), &d2, 1e-12);
    // if conditions on the block and on single updates
    s.run("gen double c = 1");
    s.run("dyngen if t > 2 {\n update c = c[_n-1] * 2 if t < 5\n}");
    assert_eq!(s.nums("c"), vec![1.0, 1.0, 2.0, 4.0, 1.0]);
    s.fails("dyngen {\n gen z = 1\n}");
}

#[test]
fn putmata_getmata_round_trip() {
    let mut s = Session::new();
    s.script(
        "set obs 5
         gen t = _n
         gen x = t * t
         gen str3 s = \"a\" + string(t)",
    );
    s.run("putmata t (X = t x) if t > 1");
    assert!(s.run("mata: rows(X)").contains('4'));
    s.run("putmata s");
    s.run("getmata (a b) = X, force");
    assert_eq!(&s.nums("b")[..4], &[4.0, 9.0, 16.0, 25.0]);
    assert!(s.nums("b")[4].is_nan());
    s.run("getmata s2 = s");
    assert_eq!(s.strs("s2"), s.strs("s"));
    s.fails("getmata s2 = s");
    s.run("getmata s2 = s, replace");
    s.run("putmata xx = x, omitmissing");
    s.run("mata: xx = xx :+ 1");
    s.run("getmata x = xx, replace");
    assert_eq!(s.nums("x"), vec![2.0, 5.0, 10.0, 17.0, 26.0]);
    s.fails("getmata (p q) = xx");
}
