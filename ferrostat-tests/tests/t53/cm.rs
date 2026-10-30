#[allow(unused_imports)]
use ferrostat_tests::*;

/// Long-format choice data: `ncase` cases x `nalt` alternatives, utility
/// 0.5*alt2 - 0.3*alt3 (+...) + 1.0*x + 0.4*z*(alt==2) + Gumbel error.
fn choice_data(s: &mut Session, ncase: usize, nalt: usize, seed: u32) {
    s.script(&format!(
        "clear
         set obs {}
         set seed {seed}
         gen id = ceil(_n/{nalt})
         gen alt = mod(_n-1,{nalt})+1
         gen double x = rnormal()
         gen double w = rnormal()
         bysort id: gen double z = rnormal() if _n == 1
         bysort id: replace z = z[1]
         gen double u = 1.0*x + 0.5*(alt==2) - 0.3*(alt==3) + 0.2*(alt==4) + 0.4*z*(alt==2) - ln(-ln(runiform()))
         bysort id: egen double um = max(u)
         gen byte y = u == um
         gen byte a2 = alt == 2
         gen byte a3 = alt == 3
         gen byte a4 = alt == 4
         gen double za2 = z*a2
         gen double za3 = z*a3
         gen double za4 = z*a4",
        ncase * nalt
    ));
}

#[test]
fn cmclogit_matches_clogit_on_expanded_data() {
    let mut s = Session::new();
    choice_data(&mut s, 300, 3, 101);
    s.run("clogit y x a2 a3 za2 za3, group(id)");
    let bx = s.scalar("_b[x]");
    let ba2 = s.scalar("_b[a2]");
    let bz3 = s.scalar("_b[za3]");
    let ll = s.scalar("e(ll)");
    let sex = s.scalar("_se[x]");
    s.run("cmset id alt");
    s.run("cmclogit y x, casevars(z) basealternative(1)");
    assert_close(s.scalar("_b[alt:x]"), bx, 1e-6);
    assert_close(s.scalar("_b[2:_cons]"), ba2, 1e-6);
    assert_close(s.scalar("_b[3:z]"), bz3, 1e-6);
    assert_close(s.scalar("e(ll)"), ll, 1e-6);
    assert_close(s.scalar("_se[alt:x]"), sex, 1e-5);
    assert_eq!(s.scalar("e(N_case)"), 300.0);
    assert_eq!(s.scalar("e(k_alt)"), 3.0);
    // predicted probabilities sum to one within case and match clogit
    s.run("predict p");
    s.run("bysort id: egen double ps = total(p)");
    s.run("summarize ps");
    assert_close(s.scalar("r(min)"), 1.0, 1e-9);
    assert_close(s.scalar("r(max)"), 1.0, 1e-9);
    s.run("predict xb, xb");
    s.run("estat alternatives");
    // replay
    let out = s.run("cmclogit");
    assert!(out.contains("Conditional logit choice model"), "{out}");
}

#[test]
fn cmrologit_with_only_top_rank_equals_cmclogit() {
    let mut s = Session::new();
    choice_data(&mut s, 200, 3, 7);
    s.run("cmset id alt");
    s.run("cmclogit y x, noconstant");
    let b = s.scalar("_b[alt:x]");
    let ll = s.scalar("e(ll)");
    s.run("cmrologit y x, noconstant");
    assert_close(s.scalar("_b[alt:x]"), b, 1e-6);
    assert_close(s.scalar("e(ll)"), ll, 1e-6);
    // rologit with group() and plain names
    s.run("rologit y x, group(id)");
    assert_close(s.scalar("_b[x]"), b, 1e-6);
    // full rankings: rank by utility, larger is better
    s.run("bysort id (u): gen rk = _n");
    s.run("cmrologit rk x, noconstant");
    assert!(s.scalar("_b[alt:x]") > 0.5);
    // reverse: smaller values preferred gives the opposite sign
    s.run("cmrologit rk x, noconstant reverse");
    assert!(s.scalar("_b[alt:x]") < -0.5);
}

#[test]
fn cmmixlogit_without_heterogeneity_is_close_to_cmclogit() {
    let mut s = Session::new();
    choice_data(&mut s, 300, 3, 11);
    s.run("cmset id alt");
    s.run("cmclogit y x w");
    let bx = s.scalar("_b[alt:x]");
    let ll = s.scalar("e(ll)");
    s.run("cmmixlogit y x, random(w) intpoints(50)");
    assert!((s.scalar("_b[alt:x]") - bx).abs() < 0.1, "{} vs {bx}", s.scalar("_b[alt:x]"));
    assert!(s.scalar("e(ll)") >= ll - 1e-6);
    assert!(s.scalar("e(ll)") - ll < 2.0);
    assert_eq!(s.scalar("e(n_sim)"), 50.0);
    s.run("predict pm");
    s.run("bysort id: egen double ps = total(pm)");
    s.run("summarize ps");
    assert_close(s.scalar("r(mean)"), 1.0, 1e-9);
}

#[test]
fn cmxtmixlogit_on_panel_choice_data() {
    let mut s = Session::new();
    choice_data(&mut s, 120, 3, 5);
    s.run("gen pid = ceil(id/4)");
    s.run("bysort pid id: gen t = id - 4*(pid-1)");
    s.run("replace t = id - 4*(pid-1)");
    s.run("cmset pid t alt");
    assert!(s.has_var("_caseid"));
    s.run("cmxtmixlogit y x, random(w) intpoints(30)");
    assert_eq!(s.scalar("e(N_panel)"), 30.0);
    assert_eq!(s.scalar("e(N_case)"), 120.0);
    assert!(s.scalar("_b[alt:x]") > 0.4);
}

#[test]
fn cmmprobit_and_cmroprobit_agree_in_sign_with_logit() {
    let mut s = Session::new();
    choice_data(&mut s, 250, 3, 3);
    s.run("cmset id alt");
    s.run("cmclogit y x");
    let bl = s.scalar("_b[alt:x]");
    s.run("cmmprobit y x, intpoints(50)");
    let bp = s.scalar("_b[alt:x]");
    assert!(bp > 0.0 && bl > 0.0);
    // logistic scale is larger than the probit one
    assert!(bp < bl, "{bp} {bl}");
    assert_eq!(s.scalar("e(converged)"), 1.0);
    s.run("predict pp");
    s.run("bysort id: egen double ps = total(pp)");
    s.run("summarize ps");
    assert!((s.scalar("r(mean)") - 1.0).abs() < 0.02);
    s.run("cmmprobit y x, intpoints(50) stddev(heteroskedastic)");
    s.run("cmmprobit y x, intpoints(50) correlation(unstructured)");
    s.run("estat covariance");
    s.run("bysort id (u): gen rk = 4 - _n");
    s.run("cmroprobit rk x, intpoints(50)");
    assert!(s.scalar("_b[alt:x]") > 0.3);
}

#[test]
fn nlogit_nests_conditional_logit() {
    let mut s = Session::new();
    choice_data(&mut s, 300, 4, 21);
    s.run("cmset id alt");
    s.run("cmclogit y x, casevars(z) basealternative(1)");
    let ll = s.scalar("e(ll)");
    let bx = s.scalar("_b[alt:x]");
    s.run("nlogitgen nest = alt(left: 1|2, right: 3|4)");
    assert!(s.has_var("nest"));
    let out = s.run("nlogittree alt nest, choice(y)");
    assert!(out.contains("left") && out.contains("right"), "{out}");
    s.run("nlogit y x || nest: , base(left) || alt: z, base(1) case(id)");
    assert!(s.scalar("e(ll)") >= ll - 1e-6);
    assert_close(s.scalar("e(ll_c)"), ll, 1e-6);
    assert!((s.scalar("_b[alt:x]") - bx).abs() < 0.3);
    let t1 = s.scalar("_b[/left_tau]");
    let t2 = s.scalar("_b[/right_tau]");
    assert!((t1 - 1.0).abs() < 0.6 && (t2 - 1.0).abs() < 0.6, "{t1} {t2}");
    s.run("predict pn");
    s.run("bysort id: egen double ps = total(pn)");
    s.run("summarize ps");
    assert_close(s.scalar("r(mean)"), 1.0, 1e-9);
    s.run("predict cp, condp");
}

#[test]
fn cm_descriptive_tools_count_cases() {
    let mut s = Session::new();
    choice_data(&mut s, 50, 3, 9);
    // first ten cases lack alternative 3
    s.run("drop if id <= 10 & alt == 3");
    s.run("bysort id: egen double um2 = max(u)");
    s.run("replace y = u == um2");
    s.fails("cmtab, choice(y)");
    s.run("cmset id alt");
    s.run("cmtab, choice(y)");
    assert_eq!(s.scalar("r(N)"), 50.0);
    s.run("cmchoiceset, generate(cs)");
    assert_eq!(s.scalar("r(N)"), 50.0);
    assert_eq!(s.scalar("r(r)"), 2.0);
    s.run("cmchoiceset, size");
    s.run("cmsummarize x, choice(y)");
    s.run("replace x = . in 1");
    s.run("cmsample x, choice(y) generate(why)");
    // the whole first case (2 rows) is excluded
    assert_eq!(s.scalar("r(N_ex)"), 2.0);
    assert_eq!(s.scalar("r(N_used)"), 138.0);
    s.run("cmclogit y x");
    assert_eq!(s.scalar("e(N_case)"), 49.0);
    let out = s.run("cmset");
    assert!(out.contains("Case ID variable"), "{out}");
    s.run("cmset, clear");
    s.fails("cmclogit y x");
}
