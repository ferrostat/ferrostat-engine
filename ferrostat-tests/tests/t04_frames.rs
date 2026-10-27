//! Data frames: several datasets in memory, links between them.

use ferrostat_tests::*;

#[test]
fn create_change_and_current_frame() {
    let mut s = Session::with_fixture();
    assert!(s.run("frame pwf").contains("default"));
    s.run("frame create other");
    s.run("frame change other");
    assert_eq!(s.n(), 0);
    assert!(s.run("display c(frame)").contains("other"));
    s.run("set obs 3");
    s.run("generate a = _n");
    s.run("cwf default");
    assert_eq!(s.n(), 12);
    assert!(s.has_var("x"));
    s.run("frame other: replace a = a * 10");
    s.run("frames change other");
    assert_eq!(s.nums("a"), vec![10.0, 20.0, 30.0]);
    s.fails("frame create other");
    s.fails("frame change nosuch");
    s.fails("frame drop other");
}

#[test]
fn create_with_variables_dir_rename_drop() {
    let mut s = Session::with_fixture();
    s.run("frame create results double b str10 name");
    s.run("frame results: set obs 1");
    s.run("frame results: replace name = \"hi\"");
    let d = s.run("frame dir");
    assert!(d.contains("default") && d.contains("results") && d.contains("12 x 8"), "{d}");
    s.run("frame rename results res2");
    s.fails("frame change results");
    s.run("frame res2: assert name == \"hi\"");
    s.run("frame drop res2");
    s.fails("frame change res2");
}

#[test]
fn copy_put_and_reset() {
    let mut s = Session::with_fixture();
    s.run("frame copy default backup");
    s.run("drop if group == 1");
    let n = s.n();
    s.run("frame backup: count");
    assert_eq!(s.scalar("r(N)"), 12.0);
    s.fails("frame copy default backup");
    s.run("frame copy default backup, replace");
    s.run("frame backup: count");
    assert_eq!(s.scalar("r(N)"), n as f64);
    s.run("frame put id x if x > 5, into(sub)");
    s.run("cwf sub");
    assert_eq!(s.varnames(), vec!["id", "x"]);
    assert!(s.nums("x").iter().all(|&v| v > 5.0));
    s.run("frame reset");
    assert_eq!(s.n(), 0);
    assert!(s.run("pwf").contains("default"));
    s.fails("frame change backup");
}

#[test]
fn clear_all_drops_frames() {
    let mut s = Session::with_fixture();
    s.run("frame create extra");
    s.run("cwf extra");
    s.run("clear all");
    assert!(s.run("display c(frame)").contains("default"));
    s.fails("cwf extra");
}

#[test]
fn preserve_is_per_frame() {
    let mut s = Session::with_fixture();
    s.run("frame create f2");
    s.run("preserve");
    s.run("drop if group == 1");
    s.run("frame f2: set obs 5");
    s.run("restore");
    assert_eq!(s.n(), 12);
    s.run("frame f2: count");
    assert_eq!(s.scalar("r(N)"), 5.0);
    s.run("frame f2: capture restore");
    assert_eq!(s.scalar("_rc"), 622.0);
}

/// Two small frames: persons (m:1 to groups) and a group table.
fn linked() -> Session {
    let mut s = Session::with_fixture();
    s.script(
        "frame create grp
         frame grp: set obs 3
         frame grp: generate group = _n
         frame grp: generate size = 100 * _n
         frame grp: generate str5 gname = cond(_n == 1, \"one\", \"other\")",
    );
    s
}

#[test]
fn frlink_frget_frval() {
    let mut s = linked();
    let out = s.run("frlink m:1 group, frame(grp)");
    assert!(out.contains("matched"), "{out}");
    assert!(s.has_var("grp"));
    let g = s.nums("group");
    let link = s.nums("grp");
    for (a, b) in g.iter().zip(&link) {
        assert_eq!(a, b);
    }
    let out = s.run("frget size gname, from(grp)");
    assert!(out.contains("2 variables copied"), "{out}");
    let size = s.nums("size");
    for (a, b) in g.iter().zip(&size) {
        assert_eq!(a * 100.0, *b);
    }
    assert_eq!(s.strs("gname")[0], if g[0] == 1.0 { "one" } else { "other" });
    s.run("generate r = frval(grp, size) / group");
    assert!(s.nums("r").iter().all(|&v| v == 100.0));
    s.run("frget sz = size, from(grp)");
    s.run("assert sz == size");
    s.run("frget *, from(grp) prefix(g_)");
    assert!(s.has_var("g_size") && !s.has_var("g_group"));
    s.fails("frlink 1:1 group, frame(grp)");
}

#[test]
fn frlink_unmatched_and_keys_with_other_names() {
    let mut s = linked();
    s.run("frame grp: rename group gid");
    s.run("frame grp: drop if gid >= 2");
    let out = s.run("frlink m:1 group, frame(grp gid) generate(lnk)");
    assert!(out.contains("unmatched"), "{out}");
    let out = s.run("frget size, from(lnk)");
    assert!(out.contains("missing values generated"), "{out}");
    let g = s.nums("group");
    let size = s.nums("size");
    for (a, b) in g.iter().zip(&size) {
        if *a == 1.0 { assert_eq!(*b, 100.0) } else { assert!(b.is_nan()) }
    }
}

#[test]
fn frame_block_runs_in_other_frame() {
    let mut s = Session::with_fixture();
    s.run("frame create b");
    s.run("frame b {\nset obs 4\ngenerate v = _n\n}");
    assert!(s.run("display c(frame)").contains("default"));
    s.run("frame b: assert _N == 4");
    // the current frame is restored after an error
    s.fails("frame b: generate v = 1");
    assert!(s.run("display c(frame)").contains("default"));
}
