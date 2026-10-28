//! Natively implemented community data-management packages: gtools, ftools,
//! fre, sumup, tabout, egenmore functions, rangestat, carryforward, labutil.

use ferrostat_tests::*;

fn same_vars(a: &mut Session, b: &mut Session, vars: &[&str]) {
    assert_eq!(a.n(), b.n());
    for v in vars {
        assert_all_close(&a.nums(v), &b.nums(v), 1e-9);
    }
}

#[test]
fn gcollapse_matches_collapse() {
    let mut a = Session::with_fixture();
    let mut b = Session::with_fixture();
    a.run("gcollapse (mean) mx=x (sd) sy=y (sum) sz=z (count) n=x (max) mxz=z (median) md=y (p25) q=y, by(group)");
    b.run("collapse (mean) mx=x (sd) sy=y (sum) sz=z (count) n=x (max) mxz=z (median) md=y (p25) q=y, by(group)");
    same_vars(&mut a, &mut b, &["group", "mx", "sy", "sz", "n", "mxz", "md", "q"]);
    let mut c = Session::with_fixture();
    c.run("fcollapse (mean) x [aw=z], by(group)");
    let mut d = Session::with_fixture();
    d.run("collapse (mean) x [aw=z], by(group)");
    same_vars(&mut c, &mut d, &["x"]);
}

#[test]
fn gcollapse_extras_merge_freq() {
    let mut s = Session::with_missing_fixture();
    s.run("gcollapse (nunique) nu=z (nmissing) nm=z (geomean) gm=y (first) f=name (last) l=name, by(group) freq(nobs)");
    assert!(s.has_var("nobs") && s.has_var("f"));
    let n = s.nums("nobs");
    assert_eq!(n.iter().sum::<f64>(), 12.0);
    assert_eq!(s.nums("nm").iter().sum::<f64>(), 1.0);
    let mut m = Session::with_fixture();
    m.run("gcollapse (mean) mx=x, by(group) merge freq(k)");
    assert_eq!(m.n(), 12);
    let mut ref_ = Session::with_fixture();
    ref_.run("egen rx = mean(x), by(group)");
    assert_all_close(&m.nums("mx"), &ref_.nums("rx"), 1e-9);
    assert!(m.nums("k").iter().all(|&k| k >= 1.0));
    let mut l = Session::with_fixture();
    l.run(r#"gcollapse (mean) x, by(group) labelformat(#stat# of #sourcelabel#)"#);
    assert!(l.has_var("x"));
    l.fails("gcollapse (bogus) x, by(group)");
}

#[test]
fn gegen_and_extras() {
    let mut a = Session::with_fixture();
    let mut b = Session::with_fixture();
    a.run("gegen m = mean(y), by(group)");
    b.run("egen m = mean(y), by(group)");
    same_vars(&mut a, &mut b, &["m"]);
    a.run("gegen nu = nunique(z), by(group)");
    a.run("gegen p = p75(y), by(group)");
    a.run("gegen fz = first(z), by(group)");
    a.run("fegen g = group(group d)");
    assert!(a.nums("g").iter().all(|&x| x >= 1.0));
    // brute-force distinct count of z within group
    let (g, z) = (a.nums("group"), a.nums("z"));
    for i in 0..12 {
        let mut s: Vec<f64> = (0..12).filter(|&j| g[j] == g[i]).map(|j| z[j]).collect();
        s.sort_by(f64::total_cmp);
        s.dedup();
        assert_eq!(a.nums("nu")[i], s.len() as f64);
    }
    a.run("gegen nu = nunique(z), by(group) replace");
    a.fails("gegen nu = nunique(z), by(group)");
}

#[test]
fn egenmore_functions() {
    let mut s = Session::with_fixture();
    s.run("egen nv = nvals(z)");
    assert_eq!(s.nums("nv")[0], 8.0);
    s.run("egen ra = rall(x y z), cond(@ > 3)");
    s.run("egen rn = rany(x y z), cond(@ > 15)");
    s.run("egen rc = rcount(x y z), cond(@ > 3)");
    let (x, y, z) = (s.nums("x"), s.nums("y"), s.nums("z"));
    for i in 0..12 {
        let k = [x[i], y[i], z[i]].iter().filter(|&&v| v > 3.0).count();
        assert_eq!(s.nums("rc")[i], k as f64);
        assert_eq!(s.nums("ra")[i], (k == 3) as u8 as f64);
        assert_eq!(s.nums("rn")[i], [x[i], y[i], z[i]].iter().any(|&v| v > 15.0) as u8 as f64);
    }
    s.run("egen rec = record(z)");
    let mut best = f64::MIN;
    for i in 0..12 {
        best = best.max(z[i]);
        assert_eq!(s.nums("rec")[i], best);
    }
    s.run("egen out = outside(z), lower(2) upper(6)");
    assert_eq!(s.nums("out")[5], 1.0);
}

#[test]
fn rangestat_matches_brute_force() {
    let mut s = Session::with_fixture();
    s.run("rangestat (mean) y (count) y (sd) z (max) z, interval(id -2 1)");
    let (y, z) = (s.nums("y"), s.nums("z"));
    for i in 0..12 {
        let lo = (i as usize).saturating_sub(2);
        let hi = (i as usize + 1).min(11);
        let w: Vec<f64> = (lo..=hi).map(|j| y[j]).collect();
        assert_close(s.nums("y_mean")[i], w.iter().sum::<f64>() / w.len() as f64, 1e-9);
        assert_eq!(s.nums("y_count")[i], w.len() as f64);
        let zw: Vec<f64> = (lo..=hi).map(|j| z[j]).collect();
        assert_eq!(s.nums("z_max")[i], zw.iter().cloned().fold(f64::MIN, f64::max));
        let m = zw.iter().sum::<f64>() / zw.len() as f64;
        let sd = (zw.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (zw.len() as f64 - 1.0)).sqrt();
        assert_close(s.nums("z_sd")[i], sd, 1e-9);
    }
    // excludeself, by, unbounded
    let mut t = Session::with_fixture();
    t.run("rangestat (sum) newy=y, interval(id . 0) by(group) excludeself");
    let (g, y) = (t.nums("group"), t.nums("y"));
    for i in 0..12 {
        let b: f64 = (0..i).filter(|&j| g[j] == g[i]).map(|j| y[j]).sum();
        assert_close(t.nums("newy")[i], b, 1e-9);
    }
}

#[test]
fn carryforward_fills() {
    let mut s = Session::with_fixture();
    s.run("replace y = . in 3/4");
    s.run("replace y = . in 1");
    s.run("carryforward y, gen(yf) cfindicator(ci)");
    let yf = s.nums("yf");
    assert!(yf[0].is_nan() || yf[0] > 1e300);
    assert_eq!(yf[2], 3.9);
    assert_eq!(yf[3], 3.9);
    assert_eq!(s.nums("ci")[2], 1.0);
    assert_eq!(s.nums("ci")[1], 0.0);
    s.run("carryforward y, gen(yb) back");
    assert_eq!(s.nums("yb")[0], 3.9);
    assert_eq!(s.nums("yb")[2], 10.1);
    s.run("bysort group: carryforward z, replace");
    assert!(s.has_var("z"));
}

#[test]
fn fre_table_and_results() {
    let mut s = Session::with_missing_fixture();
    let out = s.run("fre d");
    assert!(out.contains("Valid") && out.contains("0 no") && out.contains("1 yes") && out.contains("Total"), "{out}");
    assert_eq!(s.scalar("r(N)"), 12.0);
    assert_eq!(s.scalar("r(r)"), 2.0);
    let out = s.run("fre y");
    assert!(out.contains("Missing"), "{out}");
    assert_eq!(s.scalar("r(N_miss)"), 1.0);
    assert_eq!(s.scalar("r(N_valid)"), 11.0);
}

#[test]
fn sumup_and_tabout() {
    let mut s = Session::with_fixture();
    let out = s.run("sumup x y, by(group)");
    assert!(out.contains("mean"), "{out}");
    let dir = temp_dir("t31_tabout");
    let f = dir.join("t.csv");
    let f = f.to_str().unwrap();
    let out = s.run(&format!("tabout group d using {f}, replace style(csv) cells(freq col)"));
    assert!(out.contains("Table output written"), "{out}");
    let txt = ferrostat_core::vfs::read_to_string(f).unwrap();
    assert!(txt.contains("Total"), "{txt}");
    assert_eq!(s.scalar("r(N)"), 12.0);
    let t = s.run("tabout group, style(tsv)");
    assert!(t.contains("Total\t12"), "{t}");
    let h = s.run("tabout group d, style(htm)");
    assert!(h.contains("<table"));
    s.fails("tabout group d, cells(bogus)");
}

#[test]
fn gtools_misc_commands() {
    let mut s = Session::with_fixture();
    let out = s.run("gunique group");
    assert!(out.contains("unique values of group"), "{out}");
    s.run("gunique group, generate(gid) counts(gn)");
    assert_eq!(s.scalar("r(J)"), 3.0);
    s.run("glevelsof group, local(lv)");
    assert_eq!(s.scalar("r(J)"), 3.0);
    let lv = s.run("glevelsof group d, colseparate(,) separate(;)");
    assert!(lv.contains(';') && lv.contains(','), "{lv}");
    s.run("gdistinct group");
    assert_eq!(s.scalar("r(ndistinct)"), 3.0);
    s.run("gisid id");
    s.fails("gisid group");
    s.run("hashsort -x");
    assert!(s.nums("x").windows(2).all(|w| w[0] >= w[1]));
    s.run("fsort id");
    s.run("gduplicates report group");
    s.run("gtop group");
    assert_eq!(s.scalar("r(J)"), 3.0);
    s.run("gstats winsor y, cuts(10 90) gen(yw)");
    assert!(s.has_var("yw"));
    s.run("gstats tabstat x, stat(mean)");
    s.run("gstats sum x, by(group)");
    s.run("gstats hdfe x y, absorb(group) prefix(r_)");
    let (r, g) = (s.nums("r_x"), s.nums("group"));
    for gv in [0.0, 1.0, 2.0] {
        let sm: f64 = (0..12).filter(|&i| g[i] == gv).map(|i| r[i]).sum();
        assert_close(sm, 0.0, 1e-8);
    }
}

#[test]
fn gquantiles_and_xtile() {
    let mut s = Session::with_fixture();
    s.run("gquantiles q = y, xtile nquantiles(4)");
    let q = s.nums("q");
    assert_eq!(q.len(), 12);
    assert!(q.iter().all(|&v| (1.0..=4.0).contains(&v)));
    assert!(q.windows(2).all(|w| w[0] <= w[1]));
    s.run("fasterxtile f = y, nq(2)");
    assert_eq!(s.nums("f").iter().filter(|&&v| v == 1.0).count(), 6);
    s.run("gquantiles p = y, pctile nq(4) genp(pp)");
    assert_eq!(s.nums("pp")[1], 50.0);
    s.run("gquantiles y, _pctile nq(2)");
    assert!(s.scalar("r(r1)") > 10.0 && s.scalar("r(r1)") < 14.0);
}

#[test]
fn greshape_and_join() {
    let mut s = Session::new();
    s.script("clear\ninput id x1 x2\n1 5 6\n2 7 8\nend");
    s.run("greshape long x, by(id) keys(t)");
    assert_eq!(s.n(), 4);
    s.run("greshape wide x, by(id) keys(t)");
    assert_eq!(s.n(), 2);
    let dir = temp_dir("t31_join");
    let f = dir.join("u.dta");
    let f = f.to_str().unwrap();
    let mut u = Session::new();
    u.script("clear\ninput id w\n1 100\n2 200\nend");
    u.run(&format!("save \"{f}\", replace"));
    s.run(&format!("join, from(\"{f}\") by(id)"));
    assert_eq!(s.nums("w"), vec![100.0, 200.0]);
    assert!(s.has_var("_merge"));
}

#[test]
fn labutil_commands() {
    let mut s = Session::with_fixture();
    s.run("labmask group, values(name)");
    let out = s.run("decode group, gen(gs)");
    let _ = out;
    assert_eq!(s.strs("gs")[0], s.strs("name")[0]);
    s.run("labvalch yesno, from(0 1) to(10 11)");
    s.run("label list yesno");
    s.run("labvalclone yesno yn2");
    s.run("labvarch x, prefix(New: )");
    s.fails("labvalch nosuch, from(1) to(2)");
}

#[test]
fn packages_report_native() {
    let mut s = Session::new();
    for p in ["gtools", "ftools", "fre", "sumup", "tabout", "egenmore", "rangestat", "carryforward", "labutil"] {
        let out = s.run(&format!("ssc install {p}"));
        assert!(out.contains("nothing to install"), "{p}: {out}");
    }
    assert!(s.run("ssc install gtools").contains("gcollapse"));
}
