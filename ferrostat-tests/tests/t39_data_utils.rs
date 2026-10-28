//! Data-management utilities: variable lists and lookup, generation and
//! transformation helpers, integrity checks, text-file helpers, frame files
//! and the foreign-format importers (SPSS, SAS7BDAT, SAS transport, dBase).
//!
//! The SPSS and dBase inputs are generated here; SAS7BDAT files are checked
//! against public sample files when they are present in the directory named
//! by the `FERROSTAT_SAMPLES` environment variable (skipped otherwise).

use ferrostat_tests::*;

fn tmp(name: &str) -> String {
    temp_dir(name).to_string_lossy().to_string()
}

#[test]
fn ds_lookfor_and_vl() {
    let mut s = Session::with_fixture();
    let out = s.run("ds");
    assert!(out.contains("id") && out.contains("cnt"), "{out}");
    s.run("ds, has(type string)");
    assert_eq!(s.run("display r(varlist)"), "name");
    s.run("ds, not(type string)");
    assert_eq!(s.run("display r(varlist)"), "id group x y z d cnt");
    s.run("ds, has(varl *outcome*)");
    assert_eq!(s.run("display r(varlist)"), "d");
    s.run("ds, has(vall yesno)");
    assert_eq!(s.run("display r(varlist)"), "d");
    s.run("ds, alpha");
    assert_eq!(s.run("display r(varlist)"), "cnt d group id name x y z");
    let out = s.run("lookfor outcome");
    assert!(out.contains("Binary outcome"), "{out}");
    assert_eq!(s.run("display r(varlist)"), "d");
    // variable lists
    s.run("vl set");
    let cat = s.run("display \"$vlcategorical\"");
    assert!(cat.split_whitespace().any(|v| v == "group") && cat.split_whitespace().any(|v| v == "d"), "{cat}");
    s.run("vl create mylist = (x y)");
    assert_eq!(s.run("display \"$mylist\""), "x y");
    s.run("vl drop mylist");
    assert_eq!(s.run("display \"$mylist\""), "");
    let out = s.run("varmanage");
    assert!(out.contains("not available"), "{out}");
}

#[test]
fn labelbook_reports_problems() {
    let mut s = Session::with_fixture();
    let out = s.run("labelbook");
    assert!(out.contains("value label yesno") && out.contains("variables:  d"), "{out}");
    assert_eq!(s.run("labelbook, problems"), "  no potential problems in dataset");
    s.run("label define lonely 1 \"a\" 3 \"b\" 4 \"b\"");
    let out = s.run("labelbook, problems");
    assert!(out.contains("Not used by any variables") && out.contains("Gaps"), "{out}");
    assert_eq!(s.run("display r(notused)"), "lonely");
}

#[test]
fn range_ipolate_pctile() {
    let mut s = Session::new();
    s.script("set obs 11\nrange r 0 1");
    assert_close(s.nums("r")[3], 0.3, 1e-12);
    s.run("range r2 10 20 5");
    let r2 = s.nums("r2");
    assert_eq!(&r2[..5], &[10.0, 12.5, 15.0, 17.5, 20.0]);
    assert!(r2[5].is_nan());
    // interpolation
    s.script("gen id = _n\ngen y = id*2\nreplace y = . in 3/5\nreplace y = . in 11");
    s.run("ipolate y id, gen(yi)");
    s.run("ipolate y id, gen(ye) epolate");
    let yi = s.nums("yi");
    assert_close(yi[3], 8.0, 1e-5);
    assert!(yi[10].is_nan(), "no extrapolation without epolate");
    assert_close(s.nums("ye")[10], 22.0, 1e-5);
    // percentiles
    s.script("gen v = _n");
    s.run("pctile q = v, nq(4) genp(pp)");
    let q = s.nums("q");
    assert_eq!(&q[..3], &[3.0, 6.0, 9.0]);
    assert!(q[3].is_nan());
    assert_eq!(&s.nums("pp")[..3], &[25.0, 50.0, 75.0]);
    s.run("_pctile v, percentiles(10 50 90)");
    assert_close(s.scalar("r(r2)"), 6.0, 1e-9);
}

#[test]
fn stack_and_expandcl() {
    let mut s = Session::new();
    s.script("set obs 3\ngen a1 = _n\ngen a2 = _n*10\ngen b1 = -_n\ngen b2 = -_n*10");
    s.run("stack a1 a2 b1 b2, into(x y) clear");
    assert_eq!(s.n(), 6);
    assert_eq!(s.nums("x"), vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0]);
    assert_eq!(s.nums("y"), vec![10.0, 20.0, 30.0, -10.0, -20.0, -30.0]);
    assert_eq!(s.nums("_stack"), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    let mut s = Session::new();
    s.script("set obs 6\ngen cl = ceil(_n/2)\ngen v = _n");
    s.run("expandcl 3, cluster(cl) generate(ncl)");
    assert_eq!(s.n(), 18);
    s.run("duplicates report ncl");
    let ids = s.nums("ncl");
    let mut u: Vec<i64> = ids.iter().map(|x| *x as i64).collect();
    u.sort();
    u.dedup();
    assert_eq!(u.len(), 9, "3 clusters x 3 copies");
}

#[test]
fn drawnorm_and_corr2data() {
    let mut s = Session::new();
    s.script(
        "set seed 42\nmatrix C = (1, .6 \\ .6, 1)\nmatrix m = (5, -2)\nmatrix sd = (2, 3)\ncorr2data a b, n(300) means(m) sds(sd) corr(C) double clear",
    );
    assert_eq!(s.n(), 300);
    s.run("quietly correlate a b, covariance");
    let out = s.run("quietly summarize a");
    let _ = out;
    assert_close(s.scalar("r(mean)"), 5.0, 1e-9);
    assert_close(s.scalar("r(sd)"), 2.0, 1e-9);
    s.run("quietly correlate a b");
    assert_close(s.scalar("r(rho)"), 0.6, 1e-9);
    // lower-triangle storage
    s.script("matrix L = (1, .5, 1)\ncorr2data p q, n(80) corr(L) cstorage(lower) clear");
    s.run("quietly correlate p q");
    assert_close(s.scalar("r(rho)"), 0.5, 1e-9);
    // random draws: right size and roughly right moments
    s.script("drawnorm e1 e2, n(4000) clear");
    s.run("quietly summarize e1");
    assert!(s.scalar("r(mean)").abs() < 0.08 && (s.scalar("r(sd)") - 1.0).abs() < 0.08);
    s.fails("drawnorm e3, n(10)");
}

#[test]
fn splitsample_assigns_groups() {
    let mut s = Session::new();
    s.script("set seed 7\nset obs 100\ngen g = mod(_n, 4)");
    s.run("splitsample, generate(sp) split(.5 .25 .25)");
    let sp = s.nums("sp");
    let count = |k: f64| sp.iter().filter(|&&v| v == k).count();
    assert_eq!((count(1.0), count(2.0), count(3.0)), (50, 25, 25));
    s.run("splitsample, generate(sb) nsplit(2) balance(g)");
    let sb = s.nums("sb");
    let g = s.nums("g");
    for k in 0..4 {
        let n1 = (0..100).filter(|&i| g[i] == k as f64 && sb[i] == 1.0).count();
        assert!((12..=13).contains(&n1), "stratum {k}: {n1}");
    }
    s.run("splitsample, generate(sc) nsplit(2) cluster(g)");
    let sc = s.nums("sc");
    for k in 0..4 {
        let vals: Vec<f64> = (0..100).filter(|&i| g[i] == k as f64).map(|i| sc[i]).collect();
        assert!(vals.iter().all(|v| *v == vals[0]), "clusters stay together");
    }
}

#[test]
fn compare_signature_nested() {
    let mut s = Session::with_fixture();
    s.script("gen w = x\nreplace w = w + 1 in 1/3\nreplace w = . in 12");
    let out = s.run("compare x w");
    assert!(out.contains("x<w") && out.contains("Jointly defined"), "{out}");
    assert_eq!(s.scalar("r(n_lt)"), 3.0);
    assert_eq!(s.scalar("r(n_eq)"), 8.0);
    // data signature
    s.run("datasignature set");
    s.run("datasignature confirm");
    s.run("replace y = y + 1 in 2");
    s.fails("datasignature confirm");
    s.run("datasignature set, reset");
    s.run("datasignature confirm");
    // nesting
    s.run("assertnested id group");
    s.fails("assertnested group id");
}

#[test]
fn text_file_helpers() {
    let dir = tmp("textfiles");
    let mut s = Session::with_fixture();
    s.run(&format!("outfile id name x using {dir}/o.raw, comma replace"));
    let txt = std::fs::read_to_string(format!("{dir}/o.raw")).unwrap();
    assert!(txt.starts_with("1,\"ant\",1\n"), "{txt}");
    assert_eq!(txt.lines().count(), 12);
    s.fails(&format!("outfile id using {dir}/o.raw, comma"));
    s.run(&format!("outfile id name using {dir}/d.raw, dictionary replace wide"));
    let txt = std::fs::read_to_string(format!("{dir}/d.raw")).unwrap();
    assert!(txt.starts_with("infile dictionary {") && txt.contains("str4"), "{txt}");
    s.run(&format!("outfile d using {dir}/l.raw, replace wide"));
    assert!(std::fs::read_to_string(format!("{dir}/l.raw")).unwrap().starts_with("\"no\"\n\"no\"\n\"yes\""));
    s.run(&format!("outfile d using {dir}/n.raw, replace wide nolabel"));
    assert!(std::fs::read_to_string(format!("{dir}/n.raw")).unwrap().starts_with("0\n0\n1"));
    // filefilter
    let out = s.run(&format!("filefilter {dir}/o.raw {dir}/f.raw, from(\",\") to(\";\")"));
    assert!(out.contains("24 replacements"), "{out}");
    assert!(std::fs::read_to_string(format!("{dir}/f.raw")).unwrap().starts_with("1;\"ant\";1\n"));
    // line endings
    s.run(&format!("changeeol {dir}/o.raw {dir}/w.raw, eol(windows)"));
    let w = std::fs::read(format!("{dir}/w.raw")).unwrap();
    assert!(w.windows(2).filter(|p| *p == b"\r\n").count() == 12);
    let out = s.run(&format!("hexdump {dir}/w.raw, analyze"));
    assert!(out.contains("(\\r\\n): 12"), "{out}");
    let out = s.run(&format!("hexdump {dir}/o.raw, from(0) to(8)"));
    assert!(out.contains("31 2c 22 61 6e 74 22 2c"), "{out}");
    // checksum
    let out = s.run(&format!("checksum {dir}/o.raw"));
    assert!(out.contains("size = "), "{out}");
    assert!(s.scalar("r(checksum)") > 0.0);
}

#[test]
fn zip_roundtrip_and_unicode() {
    let dir = tmp("zipuni");
    std::fs::write(format!("{dir}/a.txt"), "alpha\n").unwrap();
    std::fs::write(format!("{dir}/b.txt"), "bravo\n").unwrap();
    // latin-1 text with an e-acute
    std::fs::write(format!("{dir}/l1.txt"), b"caf\xe9\n").unwrap();
    let mut s = Session::new();
    s.run(&format!("zipfile {dir}/a.txt {dir}/b.txt, saving({dir}/arch, replace)"));
    assert!(std::path::Path::new(&format!("{dir}/arch.zip")).exists());
    std::fs::remove_file(format!("{dir}/a.txt")).unwrap();
    std::fs::remove_file(format!("{dir}/b.txt")).unwrap();
    s.run(&format!("unzipfile {dir}/arch.zip"));
    assert_eq!(std::fs::read_to_string(format!("{dir}/a.txt")).unwrap(), "alpha\n");
    assert_eq!(std::fs::read_to_string(format!("{dir}/b.txt")).unwrap(), "bravo\n");
    s.fails(&format!("unzipfile {dir}/arch.zip"));
    // unicode
    let out = s.run(&format!("unicode analyze {dir}/l1.txt"));
    assert!(out.contains("invalid UTF-8"), "{out}");
    assert_eq!(s.scalar("r(N_needs_translation)"), 1.0);
    s.run(&format!("unicode translate {dir}/l1.txt"));
    assert_eq!(std::fs::read_to_string(format!("{dir}/l1.txt")).unwrap(), "caf\u{e9}\n");
    s.run(&format!("unicode convertfile {dir}/l1.txt {dir}/back.txt, dstencoding(latin1) srcencoding(utf8)"));
    assert_eq!(std::fs::read(format!("{dir}/back.txt")).unwrap(), b"caf\xe9\n");
    let out = s.run("unicode locale list");
    assert!(out.contains("en_US"), "{out}");
    assert!(s.run("unicode collator list").contains("root"));
}

#[test]
fn frames_save_and_use() {
    let dir = tmp("frames");
    let mut s = Session::with_fixture();
    s.script("frame put id x if x > 6, into(big)\nframes describe");
    let out = s.run("frames describe big");
    assert!(out.contains("Frame big") && out.contains("Observations:"), "{out}");
    s.run(&format!("frames save {dir}/all.dtas, replace"));
    s.fails(&format!("frames save {dir}/all.dtas"));
    s.run("frames reset");
    assert_eq!(s.n(), 0);
    s.run(&format!("frames use {dir}/all.dtas, clear"));
    assert_eq!(s.n(), 12);
    let out = s.run("frames dir");
    assert!(out.contains("big") && out.contains("6 x 2"), "{out}");
    s.run("frame big: count");
    assert_eq!(s.scalar("r(N)"), 6.0);
}

#[test]
fn sas_transport_roundtrip() {
    let dir = tmp("xport");
    let mut s = Session::with_fixture();
    s.script("gen double dt = mdy(3,4,2020) + _n\nformat dt %td\nreplace z = .c in 2\nreplace z = . in 3\nlabel variable x \"The x\"");
    s.run(&format!("export sasxport5 id name x z dt using {dir}/t5.xpt, replace"));
    let raw = std::fs::read(format!("{dir}/t5.xpt")).unwrap();
    assert!(raw.starts_with(b"HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!") && raw.len() % 80 == 0);
    s.fails(&format!("export sasxport5 id using {dir}/t5.xpt"));
    s.run("clear");
    s.run(&format!("import sasxport5 using {dir}/t5.xpt"));
    assert_eq!(s.n(), 12);
    assert_eq!(s.varnames(), vec!["id", "name", "x", "z", "dt"]);
    assert_eq!(s.strs("name")[..3], ["ant", "bee", "cat"]);
    assert_eq!(s.nums("x"), X.to_vec());
    let z = s.nums("z");
    assert!(z[1].is_nan() && z[2].is_nan() && z[3] == 1.0);
    assert_eq!(s.run("display z[2]"), ".c");
    assert_eq!(s.ds().get_var("dt").unwrap().format, "%td");
    assert_eq!(s.ds().get_var("x").unwrap().label, "The x");
    assert_eq!(s.nums("dt")[0], s.scalar("mdy(3,4,2020)+1"));
    // version 8 with long names
    s.script("clear\nset obs 3\ngen a_rather_long_variable_name = _n*1.25\nlabel variable a_rather_long_variable_name \"a label that is certainly longer than forty characters\"");
    s.fails(&format!("export sasxport5 using {dir}/long.xpt"));
    s.run(&format!("export sasxport8 using {dir}/long8.xpt, replace"));
    s.run("clear");
    s.run(&format!("import sasxport8 using {dir}/long8.xpt"));
    assert!(s.has_var("a_rather_long_variable_name"));
    assert_eq!(s.nums("a_rather_long_variable_name"), vec![1.25, 2.5, 3.75]);
    assert!(s.ds().get_var("a_rather_long_variable_name").unwrap().label.ends_with("forty characters"));
}

/// A tiny SPSS system file: numeric `NUM` (value labels, long name
/// `income`) and an 8-byte string `STR` (long name `city`).
fn build_sav(compressed: bool) -> Vec<u8> {
    let mut b: Vec<u8> = Vec::new();
    let w32 = |b: &mut Vec<u8>, v: i32| b.extend(v.to_le_bytes());
    let fixed = |b: &mut Vec<u8>, s: &str, n: usize| {
        let mut v = s.as_bytes().to_vec();
        v.resize(n, b' ');
        b.extend(v);
    };
    b.extend(b"$FL2");
    fixed(&mut b, "@(#) test file", 60);
    w32(&mut b, 2);
    w32(&mut b, 2);
    w32(&mut b, if compressed { 1 } else { 0 });
    w32(&mut b, 0);
    w32(&mut b, 3);
    b.extend(100.0f64.to_le_bytes());
    fixed(&mut b, "01 Jan 20", 9);
    fixed(&mut b, "00:00:00", 8);
    fixed(&mut b, "test data", 64);
    b.extend([0u8; 3]);
    // NUM
    w32(&mut b, 2);
    w32(&mut b, 0);
    w32(&mut b, 1);
    w32(&mut b, 0);
    w32(&mut b, (5 << 16) | (8 << 8) | 2);
    w32(&mut b, (5 << 16) | (8 << 8) | 2);
    fixed(&mut b, "NUM", 8);
    w32(&mut b, 5);
    fixed(&mut b, "Score", 8);
    // STR
    w32(&mut b, 2);
    w32(&mut b, 8);
    w32(&mut b, 0);
    w32(&mut b, 0);
    w32(&mut b, (1 << 16) | (8 << 8));
    w32(&mut b, (1 << 16) | (8 << 8));
    fixed(&mut b, "STR", 8);
    // value labels for slot 1
    w32(&mut b, 3);
    w32(&mut b, 2);
    for (v, l) in [(1.0f64, "low"), (2.0, "high")] {
        b.extend(v.to_le_bytes());
        b.push(l.len() as u8);
        let mut t = l.as_bytes().to_vec();
        t.resize(7, b' ');
        b.extend(t);
    }
    w32(&mut b, 4);
    w32(&mut b, 1);
    w32(&mut b, 1);
    // long names
    let names = b"NUM=income\tSTR=city";
    w32(&mut b, 7);
    w32(&mut b, 13);
    w32(&mut b, 1);
    w32(&mut b, names.len() as i32);
    b.extend(names);
    w32(&mut b, 999);
    w32(&mut b, 0);
    let cases: [(Option<f64>, &str); 3] = [(Some(1.0), "abc"), (Some(2.0), "hello"), (None, "")];
    if compressed {
        let mut ctrl = Vec::new();
        let mut lit = Vec::new();
        for (n, s) in cases {
            match n {
                Some(v) => ctrl.push((v + 100.0) as u8),
                None => ctrl.push(255),
            }
            if s.is_empty() {
                ctrl.push(254);
            } else {
                ctrl.push(253);
                let mut t = s.as_bytes().to_vec();
                t.resize(8, b' ');
                lit.extend(t);
            }
        }
        ctrl.push(252);
        ctrl.resize(8, 0);
        b.extend(ctrl);
        b.extend(lit);
    } else {
        for (n, s) in cases {
            b.extend(n.unwrap_or(f64::MIN).to_le_bytes());
            let mut t = s.as_bytes().to_vec();
            t.resize(8, b' ');
            b.extend(t);
        }
    }
    b
}

#[test]
fn import_spss_system_files() {
    let dir = tmp("spss");
    for compressed in [false, true] {
        let path = format!("{dir}/t{}.sav", compressed as u8);
        std::fs::write(&path, build_sav(compressed)).unwrap();
        let mut s = Session::new();
        let out = s.run(&format!("import spss using {path}"));
        assert!(out.contains("(2 vars, 3 obs)"), "{out}");
        assert_eq!(s.varnames(), vec!["income", "city"]);
        let v = s.nums("income");
        assert_eq!(&v[..2], &[1.0, 2.0]);
        assert!(v[2].is_nan());
        assert_eq!(s.strs("city"), vec!["abc", "hello", ""]);
        assert_eq!(s.ds().get_var("income").unwrap().label, "Score");
        assert_eq!(s.ds().get_var("income").unwrap().value_label_name, "income");
        assert_eq!(s.ds().datalabel, "test data");
        let l = &s.ds().value_labels["income"];
        assert_eq!(l.mapping.values().cloned().collect::<Vec<_>>(), vec!["low", "high"]);
        s.run(&format!("import spss using {path}, clear case(upper)"));
        assert_eq!(s.varnames(), vec!["INCOME", "CITY"]);
    }
    std::fs::write(format!("{dir}/bad.sav"), b"not an spss file at all").unwrap();
    Session::new().fails(&format!("import spss using {dir}/bad.sav"));
}

#[test]
fn import_dbase_tables() {
    let dir = tmp("dbf");
    let mut f: Vec<u8> = vec![3, 124, 1, 1];
    f.extend(2u32.to_le_bytes());
    f.extend((32u16 + 2 * 32 + 1).to_le_bytes());
    f.extend((1u16 + 6 + 6).to_le_bytes());
    f.extend([0u8; 20]);
    for (name, ty, len) in [("NAME", b'C', 6u8), ("VAL", b'N', 6u8)] {
        let mut d = name.as_bytes().to_vec();
        d.resize(11, 0);
        d.push(ty);
        d.extend([0u8; 4]);
        d.push(len);
        d.push(0);
        d.extend([0u8; 14]);
        f.extend(d);
    }
    f.push(0x0D);
    f.extend(format!(" {:<6}{:>6}", "ann", "12.5").bytes());
    f.extend(format!(" {:<6}{:>6}", "bob", "7").bytes());
    std::fs::write(format!("{dir}/t.dbf"), &f).unwrap();
    let mut s = Session::new();
    s.run(&format!("import dbase using {dir}/t.dbf"));
    assert_eq!(s.varnames(), vec!["name", "val"]);
    assert_eq!(s.strs("name"), vec!["ann", "bob"]);
    assert_eq!(s.nums("val"), vec![12.5, 7.0]);
}

/// Cross-check the SAS7BDAT reader on public sample files (uncompressed,
/// RLE and RDC compressed, 32/64-bit, both byte orders).
#[test]
fn import_sas7bdat_samples() {
    let Ok(dir) = std::env::var("FERROSTAT_SAMPLES") else {
        eprintln!("FERROSTAT_SAMPLES not set; skipping");
        return;
    };
    let have = |f: &str| std::path::Path::new(&format!("{dir}/{f}")).exists();
    if !have("test1.sas7bdat") {
        return;
    }
    let mut s = Session::new();
    s.run(&format!("import sas using {dir}/test1.sas7bdat, clear"));
    assert_eq!((s.n(), s.ds().n_vars()), (10, 100));
    let base: Vec<Vec<f64>> = (1..=100).map(|k| s.nums(&format!("Column{k}"))).collect();
    // the same table stored with other layouts must read back identically
    for f in ["test2.sas7bdat", "test3.sas7bdat", "test5.sas7bdat", "test9.sas7bdat", "test12.sas7bdat"] {
        if !have(f) {
            continue;
        }
        s.run(&format!("import sas using {dir}/{f}, clear"));
        assert_eq!(s.n(), 10, "{f}");
        for k in 1..=100 {
            let col = s.nums(&format!("Column{k}"));
            for (a, b) in col.iter().zip(&base[k - 1]) {
                assert!((a.is_nan() && b.is_nan()) || a == b, "{f} Column{k}");
            }
        }
    }
    if have("airline.sas7bdat") {
        s.run(&format!("import sas using {dir}/airline.sas7bdat, clear"));
        assert_eq!(s.n(), 32);
        assert_close(s.nums("Y")[0], 1.2139999866485596, 1e-12);
        assert_close(s.nums("YEAR")[0], 1948.0, 1e-3);
    }
    if have("paxraw_d_short.xpt") {
        s.run(&format!("import sasxport5 using {dir}/paxraw_d_short.xpt, clear"));
        assert_eq!((s.n(), s.ds().n_vars()), (100, 9));
    }
}
