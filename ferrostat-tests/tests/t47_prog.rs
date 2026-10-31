//! Programming commands: byable/sortpreserve programs, properties, trace,
//! version, file I/O, postfile/post, frame post, sersets, sysdir/adopath/
//! findfile, the class system, mark/markin, unab family, fvexpand,
//! _rmcoll/_rmdcoll, makecns/matcproc, estimation-sample signatures,
//! tabdisp, the remaining matrix subcommands and window/pause/plugin.

use ferrostat_tests::*;

fn grouped(s: &mut Session) {
    s.script(
        "set obs 6
         gen g = ceil(_n/2)
         gen x = _n
         gen y = 2*x + 1
         gen z = mod(_n*7, 5)",
    );
}

fn path(dir: &std::path::Path, f: &str) -> String {
    dir.join(f).to_string_lossy().replace('\\', "/")
}

#[test]
fn byable_recall_and_onecall() {
    let mut s = Session::new();
    grouped(&mut s);
    s.script(
        "program define myp, byable(recall) rclass
           syntax [varlist] [if]
           marksample touse
           quietly summarize x if `touse'
           display \"by=\" _by() \" idx=\" _byindex() \" last=\" _bylastcall() \" n1=\" _byn1() \" n2=\" _byn2() \" mean=\" r(mean)
         end",
    );
    let out = s.run("by g: myp");
    assert!(out.contains("-> g = 1"), "{out}");
    assert!(out.contains("by=1 idx=1 last=0 n1=1 n2=2 mean=1.5"), "{out}");
    assert!(out.contains("by=1 idx=3 last=1 n1=5 n2=6 mean=5.5"), "{out}");
    // outside by
    let out = s.run("myp");
    assert!(out.contains("by=0 idx=1 last=1 n1=1 n2=6 mean=3.5"), "{out}");
    s.script(
        "program define one, byable(onecall)
           display \"byvars=`_byvars' by=\" _by()
         end",
    );
    let out = s.run("by g: one");
    assert!(out.contains("byvars=g by=1"), "{out}");
    assert_eq!(out.matches("byvars").count(), 1);
    s.script("program define nob\n display 1\n end");
    let e = s.try_run("by g: nob").unwrap_err();
    assert!(e.contains("may not be combined with by") && e.contains("r(190)"), "{e}");
}

#[test]
fn sortpreserve_and_properties() {
    let mut s = Session::new();
    s.script(
        "set obs 5
         gen u = 6 - _n
         gen o = _n
         program define sp, sortpreserve properties(svyb swml)
           sort u
         end
         sp",
    );
    assert_eq!(s.nums("o"), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert!(!s.varnames().iter().any(|v| v.starts_with("__sortpreserve")));
    s.run("local p : properties sp");
    assert_eq!(s.run("display \"`p'\"").trim(), "svyb swml");
    s.run("local p : properties nonesuch");
    assert_eq!(s.run("display \"[`p']\"").trim(), "[]");
    // plugin programs load pl.plugin (absent here)
    let e = s.try_run("program define pl, plugin\nend").unwrap_err();
    assert!(e.contains("pl.plugin") && e.contains("r(601)"), "{e}");
}

#[test]
fn trace_shows_program_lines() {
    let mut s = Session::new();
    s.script(
        "program define tr
           args a
           local b = `a' + 1
           display `b'
         end
         set trace on",
    );
    let out = s.run("tr 4");
    assert!(out.contains("begin tr ---"), "{out}");
    assert!(out.contains("- local b = `a' + 1"), "{out}");
    assert!(out.contains("= local b = 4 + 1"), "{out}");
    assert!(out.contains("end tr ---"), "{out}");
    s.run("set traceexpand off");
    let out = s.run("tr 4");
    assert!(!out.contains("= local b"), "{out}");
    s.run("set tracedepth 0");
    assert!(!s.run("tr 4").contains("begin tr"));
    s.run("set trace off");
    assert_eq!(s.run("tr 1").trim(), "2");
    assert_eq!(s.run("display c(trace)").trim(), "off");
}

#[test]
fn version_prefix_and_display() {
    let mut s = Session::new();
    assert_eq!(s.run("version").trim(), "version 18.0");
    assert_eq!(s.run("version 16: display c(version)").trim(), "16");
    assert_eq!(s.scalar("c(version)"), 18.0);
    s.run("version 17");
    assert_eq!(s.run("version").trim(), "version 17.0");
}

#[test]
fn file_text_write_read_seek() {
    let dir = temp_dir("t47_file");
    let p = path(&dir, "t.txt");
    let mut s = Session::new();
    s.script(&format!(
        "tempname fh
         file open `fh' using \"{p}\", write replace
         file write `fh' \"line one\" _n
         file write `fh' \"x=\" %5.2f (1/3) _tab \"end\" _n
         file write `fh' _quote \"q\" _quote _n \"last\"
         file close `fh'"
    ));
    let text = std::fs::read_to_string(&p).unwrap();
    assert_eq!(text, "line one\nx= 0.33\tend\n\"q\"\nlast");
    s.run(&format!("file open fh using \"{p}\", read"));
    s.run("file read fh line");
    assert_eq!(s.run("display \"`line'\"").trim(), "line one");
    assert_eq!(s.scalar("r(eof)"), 0.0);
    assert_eq!(s.scalar("r(status)"), 1.0);
    for _ in 0..3 {
        s.run("file read fh line");
    }
    assert_eq!(s.run("display \"`line'\"").trim(), "last");
    assert_eq!(s.scalar("r(status)"), 2.0);
    s.run("file read fh line");
    assert_eq!(s.scalar("r(eof)"), 1.0);
    s.run("file seek fh tof");
    s.run("file seek fh query");
    assert_eq!(s.scalar("r(loc)"), 0.0);
    s.run("file read fh line");
    assert_eq!(s.run("display \"`line'\"").trim(), "line one");
    let e = s.try_run(&format!("file open fh using \"{p}\", read")).unwrap_err();
    assert!(e.contains("r(110)"), "{e}");
    s.run("file close fh");
    // existing file without replace
    let e = s.try_run(&format!("file open fh using \"{p}\", write")).unwrap_err();
    assert!(e.contains("r(602)"), "{e}");
    let e = s.try_run(&format!("file open fh using \"{}\", read", path(&dir, "none.txt"))).unwrap_err();
    assert!(e.contains("r(601)"), "{e}");
    // append
    s.run(&format!("file open fh using \"{p}\", write append"));
    s.run("file write fh _n \"more\"");
    s.run("file close _all");
    assert!(std::fs::read_to_string(&p).unwrap().ends_with("last\nmore"));
}

#[test]
fn file_binary_roundtrip() {
    let dir = temp_dir("t47_bin");
    let p = path(&dir, "t.bin");
    let mut s = Session::new();
    s.run(&format!("file open b using \"{p}\", write binary replace"));
    s.run("file write b %4b (258) %8z (1.5) %1bu (200) %2bs (-3) %5s \"abc\"");
    s.run("file set b byteorder hilo");
    s.run("file write b %4bu (1)");
    s.run("file close b");
    let bytes = std::fs::read(&p).unwrap();
    assert_eq!(bytes.len(), 4 + 8 + 1 + 2 + 5 + 4);
    assert_eq!(&bytes[bytes.len() - 4..], &[0, 0, 0, 1]);
    s.run(&format!("file open b using \"{p}\", read binary"));
    s.run("file read b %4b n1 %8z n2 %1bu n3 %2bs n4 %5s str");
    assert_eq!(s.scalar("n1"), 258.0);
    assert_eq!(s.scalar("n2"), 1.5);
    assert_eq!(s.scalar("n3"), 200.0);
    assert_eq!(s.scalar("n4"), -3.0);
    assert_eq!(s.run("display \"`str'\"").trim(), "abc");
    s.run("file set b byteorder hilo");
    s.run("file read b %4bu n5");
    assert_eq!(s.scalar("n5"), 1.0);
    s.run("file read b %4b n6");
    assert_eq!(s.scalar("r(eof)"), 1.0);
    assert!(s.run("file query").contains("binary"));
    s.run("file close b");
}

#[test]
fn postfile_post_postclose() {
    let dir = temp_dir("t47_post");
    let p = path(&dir, "pp");
    let mut s = Session::new();
    s.run(&format!("postfile ph double a str5 s b using \"{p}\", replace"));
    s.run("post ph (1) (\"x\") (2.5)");
    s.run("post ph (2) (\"hello world\") (.)");
    let e = s.try_run("post ph (\"a\") (\"b\") (1)").unwrap_err();
    assert!(e.contains("r(109)") && e.contains("expression 1"), "{e}");
    assert!(s.try_run("post ph (1) (2)").is_err());
    assert!(s.run("postutil dir").contains("ph"));
    let e = s.try_run(&format!("postfile ph a using \"{p}\", replace")).unwrap_err();
    assert!(e.contains("r(110)"), "{e}");
    s.run("postclose ph");
    assert!(s.try_run("post ph (1) (\"a\") (1)").is_err());
    s.run(&format!("use \"{p}\", clear"));
    assert_eq!(s.n(), 2);
    assert_eq!(s.nums("a"), vec![1.0, 2.0]);
    assert_eq!(s.strs("s"), vec!["x", "hello"]);
    assert!(s.nums("b")[1].is_nan() || s.nums("b")[1] > 8.9e307);
    let e = s.try_run(&format!("postfile p2 a using \"{p}\"")).unwrap_err();
    assert!(e.contains("r(602)"), "{e}");
}

#[test]
fn frame_post_adds_observations() {
    let mut s = Session::new();
    s.run("frame create f2 double v str3 w");
    s.run("frame post f2 (3) (\"ab\")");
    s.run("frame post f2 (1+3) (\"cd\")");
    s.run("frame change f2");
    assert_eq!(s.nums("v"), vec![3.0, 4.0]);
    assert_eq!(s.strs("w"), vec!["ab", "cd"]);
    assert!(s.try_run("frame post nosuch (1)").is_err());
}

#[test]
fn serset_create_use_and_file() {
    let dir = temp_dir("t47_serset");
    let p = path(&dir, "ss.bin");
    let mut s = Session::new();
    s.script(
        "set obs 4
         gen u = 5 - _n
         gen a = _n
         replace a = . in 2",
    );
    s.run("serset create a u");
    assert_eq!(s.scalar("r(N)"), 3.0);
    assert_eq!(s.scalar("r(k)"), 2.0);
    assert_eq!(s.scalar("r(id)"), 0.0);
    s.run("serset create u, sort(u)");
    assert_eq!(s.scalar("r(id)"), 1.0);
    let dirout = s.run("serset dir");
    assert!(dirout.contains("3 observations on 2 variables"), "{dirout}");
    s.run("serset summarize u");
    assert_eq!(s.scalar("r(mean)"), 2.5);
    s.run("serset set 0");
    s.run(&format!("file open sf using \"{p}\", write binary replace"));
    s.run("file sersetwrite sf");
    s.run("file close sf");
    s.run("serset clear");
    assert!(s.try_run("serset id").is_err());
    s.run(&format!("file open sf using \"{p}\", read binary"));
    s.run("file sersetread sf");
    s.run("file close sf");
    s.run("serset use, clear");
    assert_eq!(s.nums("a"), vec![1.0, 3.0, 4.0]);
    assert_eq!(s.nums("u"), vec![4.0, 2.0, 1.0]);
}

#[test]
fn sysdir_adopath_findfile_viewsource() {
    let dir = temp_dir("t47_sysdir");
    let d = path(&dir, "");
    let mut s = Session::new();
    let out = s.run("sysdir");
    assert!(out.contains("PERSONAL:") && out.contains("OLDPLACE:"), "{out}");
    s.run(&format!("sysdir set PLUS \"{}\"", d.trim_end_matches('/')));
    assert_eq!(s.run("display \"`c(sysdir_plus)'\"").trim(), d);
    s.run("local p : sysdir PLUS");
    assert_eq!(s.run("display \"`p'\"").trim(), d);
    assert!(s.run("adopath").contains("(PLUS)"));
    std::fs::create_dir_all(dir.join("m")).unwrap();
    std::fs::write(dir.join("m").join("mything.ado"), "program mything\ndisplay 42\nend\n").unwrap();
    // found in the letter subdirectory of PLUS
    let out = s.run("findfile mything.ado");
    assert!(out.ends_with("m/mything.ado"), "{out}");
    assert_eq!(s.run("display \"`r(fn)'\"").trim(), out.trim());
    assert!(s.try_run("findfile mything.ado, nodescend").is_err());
    let e = s.try_run("findfile nothere.ado").unwrap_err();
    assert!(e.contains("r(601)"), "{e}");
    assert!(s.run("viewsource mything.ado").contains("display 42"));
    // adopath + / -
    let other = temp_dir("t47_sysdir2");
    std::fs::write(other.join("zz.txt"), "z").unwrap();
    s.run(&format!("adopath + \"{}\"", path(&other, "").trim_end_matches('/')));
    assert!(s.run("findfile zz.txt").contains("zz.txt"));
    s.run(&format!("adopath - \"{}\"", path(&other, "").trim_end_matches('/')));
    assert!(s.try_run("findfile zz.txt").is_err());
    assert!(s.run("personal").contains("personal ado-directory"));
}

#[test]
fn classes_members_programs_classexit() {
    let dir = temp_dir("t47_class");
    std::fs::write(
        dir.join("counter.class"),
        "version 18\nclass counter {\n    double n = 0\n    string label = \"ctr\"\n    array hist\n    classwide:\n    double made = 0\n}\nprogram .new\n    .made = `.made' + 1\nend\nprogram .bump\n    args by\n    if \"`by'\" == \"\" local by 1\n    .n = `.n' + `by'\n    .hist[`.hist.arrnels' + 1] = `.n'\nend\nprogram .value\n    classexit `.n'\nend\nprogram .show\n    display \"`.label': `.n'\"\nend\n",
    )
    .unwrap();
    let mut s = Session::new();
    s.run(&format!("cd \"{}\"", path(&dir, "")));
    s.run(".c = .counter.new");
    s.run(".c.bump");
    s.run(".c.bump 5");
    assert_eq!(s.run(".c.show").trim(), "ctr: 6");
    assert_eq!(s.scalar("`.c.n'"), 6.0);
    assert_eq!(s.run("display \"`.c.classname' `.c.isa'\"").trim(), "counter class");
    s.run("local v `.c.value'");
    assert_eq!(s.run("display \"`v'\"").trim(), "6");
    assert_eq!(s.run("display `.c.hist.arrnels' \" \" `.c.hist[2]'").trim(), "2 6");
    s.run(".d = .counter.new");
    assert_eq!(s.scalar("`.d.made'"), 2.0);
    s.run(".c.label = \"first\"");
    assert_eq!(s.run(".c.show").trim(), "first: 6");
    s.run(".d.Declare double extra = 7");
    assert_eq!(s.scalar("`.d.extra'"), 7.0);
    s.run(".e = .c.copy");
    s.run(".e.bump");
    assert_eq!(s.run("display `.e.n' \" \" `.c.n'").trim(), "7 6");
    let e = s.try_run(".c.n = \"x\"").unwrap_err();
    assert!(e.contains("r(109)"), "{e}");
    assert!(s.run("classutil dir").contains(".e"));
    assert!(s.run("classutil describe .c").contains("instance of class counter"));
    assert!(s.run("classutil describe counter").contains(".bump"));
    assert!(s.run("classutil which counter").contains("counter.class"));
    s.run("classutil drop .e");
    assert!(!s.run("classutil dir").contains(".e "));
    // a class defined in the do-file itself
    s.script(
        "class point {
           double x = 1
           double y = 2
         }
         program .norm2
           classexit `.x'^2 + `.y'^2
         end
         .p = .point.new
         .p.x = 3
         .p.y = 4",
    );
    assert_eq!(s.scalar("`.p.norm2'"), 25.0);
    assert_eq!(s.scalar("`.p.isofclass point'"), 1.0);
    s.run(".arr = {1, \"two\", 3}");
    assert_eq!(s.run("display \"`.arr[2]' `.arr.arrnels'\"").trim(), "two 3");
    s.run(".arr.arrpush 9");
    assert_eq!(s.scalar("`.arr[4]'"), 9.0);
    assert!(s.try_run(".q = .nosuchclass.new").is_err());
    s.run("discard");
    assert_eq!(s.run("classutil dir").trim(), "");
}

#[test]
fn mark_markin_markout() {
    let mut s = Session::new();
    grouped(&mut s);
    s.run("replace z = . in 6");
    s.run("mark t1 if x > 2");
    assert_eq!(s.nums("t1"), vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    s.run("markin t1 in 1/5");
    assert_eq!(s.nums("t1"), vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0]);
    s.run("mark t2");
    s.run("markout t2 z");
    assert_eq!(s.nums("t2"), vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0]);
    s.run("gen w = mod(_n, 3)");
    s.run("mark t3 [fw=w]");
    assert_eq!(s.nums("t3"), vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
    s.run("mark t4 [fw=w], zeroweight");
    assert_eq!(s.nums("t4"), vec![1.0; 6]);
    assert!(s.try_run("mark t1").is_err());
}

#[test]
fn unab_tsunab_fvunab_unabcmd() {
    let mut s = Session::new();
    grouped(&mut s);
    s.run("unab v : x-z");
    assert_eq!(s.run("display \"`v'\"").trim(), "x y z");
    s.run("tsunab v : L(1/2).x y");
    assert_eq!(s.run("display \"`v'\"").trim(), "L.x L2.x y");
    s.run("fvunab v : i.g x");
    assert_eq!(s.run("display \"`v'\"").trim(), "i.g x");
    let e = s.try_run("unab v : L.x").unwrap_err();
    assert!(e.contains("r(101)"), "{e}");
    let e = s.try_run("tsunab v : i.g").unwrap_err();
    assert!(e.contains("r(101)"), "{e}");
    let e = s.try_run("unab v : x y z, max(2)").unwrap_err();
    assert!(e.contains("r(103)"), "{e}");
    let e = s.try_run("unab v : x, min(2)").unwrap_err();
    assert!(e.contains("r(102)"), "{e}");
    assert!(s.try_run("unab v : nosuch").is_err());
    s.run("unabcmd c : reg");
    assert_eq!(s.run("display \"`c'\"").trim(), "regress");
    s.run("unabcmd c : su");
    assert_eq!(s.run("display \"`c'\"").trim(), "summarize");
    let e = s.try_run("unabcmd c : notacommand").unwrap_err();
    assert!(e.contains("r(199)"), "{e}");
}

#[test]
fn fvexpand_lists_levels() {
    let mut s = Session::new();
    grouped(&mut s);
    s.run("fvexpand i.g");
    assert_eq!(s.run("display \"`r(varlist)'\"").trim(), "1b.g 2.g 3.g");
    assert_eq!(s.run("display \"`r(fvops)'\"").trim(), "true");
    s.run("fvexpand i.g if g > 1");
    assert_eq!(s.run("display \"`r(varlist)'\"").trim(), "2b.g 3.g");
    s.run("fvexpand x-z");
    assert_eq!(s.run("display \"`r(varlist)'\"").trim(), "x y z");
    assert_eq!(s.run("display \"[`r(fvops)']\"").trim(), "[]");
}

#[test]
fn rmcoll_and_rmdcoll() {
    let mut s = Session::new();
    grouped(&mut s);
    s.run("gen w = 3*x");
    let out = s.run("_rmcoll x w z");
    assert!(out.contains("note: w omitted because of collinearity."), "{out}");
    assert_eq!(s.run("display \"`r(varlist)'\"").trim(), "x o.w z");
    assert_eq!(s.scalar("r(k_omitted)"), 1.0);
    s.run("_rmcoll x w z, forcedrop");
    assert_eq!(s.run("display \"`r(varlist)'\"").trim(), "x z");
    s.run("_rmcoll x z");
    assert_eq!(s.scalar("r(k_omitted)"), 0.0);
    let e = s.try_run("_rmdcoll y x z").unwrap_err();
    assert!(e.contains("r(459)"), "{e}");
    s.run("replace y = y + mod(_n*_n, 7)/3");
    s.run("_rmdcoll y x z");
    assert_eq!(s.run("display \"`r(varlist)'\"").trim(), "x z");
}

#[test]
fn makecns_matcproc() {
    let mut s = Session::new();
    grouped(&mut s);
    s.run("replace y = y + z/3 + mod(_n,2)");
    s.run("regress y x z");
    s.run("constraint 1 x = 1");
    s.run("constraint 2 x + z = 0");
    s.run("constraint 3 2*x = 2");
    let out = s.run("makecns 1-3, displaycns");
    assert!(out.contains("constraint number 3 dropped"), "{out}");
    assert!(out.contains("x + z = 0"), "{out}");
    assert_eq!(s.scalar("r(k)"), 2.0);
    assert_eq!(s.run("display \"`r(clist)'\"").trim(), "1 2");
    s.run("matrix C = e(Cns)");
    assert_eq!(s.scalar("rowsof(C)"), 2.0);
    assert_eq!(s.scalar("colsof(C)"), 4.0);
    assert_eq!(s.scalar("C[2,2]"), 1.0);
    assert_eq!(s.scalar("C[1,4]"), 1.0);
    s.run("matcproc T a C");
    // b = T*f + a' satisfies the constraints for any f
    assert_eq!(s.scalar("colsof(T)"), 1.0);
    assert_close(s.scalar("a[1,1]"), 1.0, 1e-10);
    assert_close(s.scalar("a[1,2]"), -1.0, 1e-10);
    assert_close(s.scalar("T[1,1]"), 0.0, 1e-10);
    assert_close(s.scalar("T[2,1]"), 0.0, 1e-10);
    s.run("makecns 1, r");
    assert_eq!(s.scalar("rowsof(r(Cns))"), 1.0);
}

#[test]
fn estimation_sample_signature() {
    let mut s = Session::new();
    grouped(&mut s);
    assert!(s.try_run("checkestimationsample").is_err());
    s.run("regress y x z");
    s.run("signestimationsample y x z");
    assert!(!s.run("display \"`e(datasignature)'\"").trim().is_empty());
    assert_eq!(s.run("display \"`e(datasignaturevars)'\"").trim(), "y x z");
    s.run("checkestimationsample");
    s.run("replace g = 9");
    s.run("checkestimationsample");
    s.run("replace y = 0 in 1");
    let e = s.try_run("checkestimationsample").unwrap_err();
    assert!(e.contains("r(459)"), "{e}");
}

#[test]
fn tabdisp_tables() {
    let mut s = Session::new();
    s.script(
        "input g b y
         1 0 1.5
         1 1 2.25
         2 0 3
         2 1 4.125
         3 0 5
         end
         label define gl 1 one 2 two 3 three
         label values g gl",
    );
    let e = s.try_run("tabdisp g, cell(y)").unwrap_err();
    assert!(e.contains("r(459)"), "{e}");
    let out = s.run("tabdisp g if b==0, cell(y)");
    assert!(out.lines().any(|l| l.trim_start().starts_with("two |") && l.trim_end().ends_with('3')), "{out}");
    let out = s.run("tabdisp g b, cell(y) format(%6.2f)");
    assert!(out.contains("1.50") && out.contains("4.12"), "{out}");
    let three = out.lines().find(|l| l.contains("three")).unwrap();
    assert!(three.trim_end().ends_with("5.00"), "{out}");
    let out = s.run("tabdisp g b, cell(y b)");
    assert!(out.lines().filter(|l| l.contains('|')).count() >= 8, "{out}");
}

#[test]
fn matrix_get_score_svd_eigen_joins() {
    let mut s = Session::new();
    grouped(&mut s);
    s.run("replace y = y + mod(_n*_n, 7)/3");
    s.run("regress y x z");
    s.run("matrix b = get(_b)");
    assert_close(s.scalar("b[1,1]"), s.scalar("_b[x]"), 1e-12);
    s.run("matrix V = get(VCE)");
    assert_close(s.scalar("sqrt(V[2,2])"), s.scalar("_se[z]"), 1e-12);
    s.run("matrix se = get(_se)");
    assert_close(s.scalar("se[1,1]"), s.scalar("_se[x]"), 1e-12);
    s.run("predict double xb");
    s.run("matrix score double sc = b");
    let (a, c) = (s.nums("xb"), s.nums("sc"));
    for i in 0..6 {
        assert_close(c[i], a[i], 1e-10);
    }
    assert!(s.try_run("matrix score sc = b").is_err());
    s.run("matrix score sc = b if x > 3, replace");
    s.run("matrix A = (2,1,0\\1,3,1\\0,1,4)");
    s.run("matrix svd U w V = A");
    s.run("matrix C = U*diag(w)*V'");
    for i in 1..=3 {
        for j in 1..=3 {
            assert_close(s.scalar(&format!("C[{i},{j}]")), s.scalar(&format!("A[{i},{j}]")), 1e-10);
        }
    }
    assert!(s.scalar("w[1,1]") >= s.scalar("w[1,2]"));
    s.run("matrix B = (0,-1\\1,0)");
    s.run("matrix eigenvalues re im = B");
    assert_close(s.scalar("re[1,1]"), 0.0, 1e-12);
    assert_close(s.scalar("abs(im[1,1])"), 1.0, 1e-12);
    s.run("matrix S = (2,1\\1,2)");
    s.run("matrix eigenvalues re im = S");
    assert_close(s.scalar("re[1,1]"), 3.0, 1e-12);
    assert_close(s.scalar("re[1,2]"), 1.0, 1e-12);
    s.run("matrix R1 = (1,2)");
    s.run("matrix colnames R1 = a b");
    s.run("matrix R2 = (3,4)");
    s.run("matrix colnames R2 = b c");
    s.run("matrix rowjoinbyname J = R1 R2");
    assert_eq!(s.scalar("colsof(J)"), 3.0);
    assert_eq!(s.scalar("J[2,2]"), 3.0);
    assert_eq!(s.scalar("missing(J[1,3])"), 1.0);
    s.run("matrix rowjoinbyname J0 = R1 R2, missing(0)");
    assert_eq!(s.scalar("J0[1,3]"), 0.0);
    s.run("matrix coljoinbyname K = R1' R2'");
    assert_eq!(s.scalar("rowsof(K)"), 3.0);
    assert_eq!(s.scalar("K[3,2]"), 4.0);
}

#[test]
fn settings_window_pause_discard_plugins() {
    let mut s = Session::new();
    s.run("set sortmethod qsort");
    assert_eq!(s.run("display c(sortmethod)").trim(), "qsort");
    assert!(s.try_run("set sortmethod nonsense").is_err());
    s.run("set sortrngstate off");
    assert_eq!(s.run("display c(sortrngstate)").trim(), "off");
    s.run("window manage forward results");
    s.run("window menu clear");
    s.run("window push regress y x");
    s.run("window fopen fn \"Open\" \"*.dta\"");
    assert_eq!(s.run("display \"[`fn']\"").trim(), "[]");
    assert!(s.run("window stopbox note \"hello\"").contains("hello"));
    let e = s.try_run("window stopbox stop \"bad\"").unwrap_err();
    assert!(e.contains("r(1)"), "{e}");
    assert_eq!(s.run("pause").trim(), "");
    s.run("pause on");
    assert!(s.run("pause check").contains("pause:"));
    s.run("pause off");
    grouped(&mut s);
    s.run("regress y x");
    s.run("discard");
    assert_eq!(s.scalar("missing(e(N))"), 1.0);
    let e = s.try_run("plugin call myplug").unwrap_err();
    assert!(e.contains("not found") && e.contains("r(199)"), "{e}");
    let e = s.try_run("javacall a b").unwrap_err();
    assert!(e.contains("jars()"), "{e}");
}
