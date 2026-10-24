//! Creating, modifying, subsetting, combining and restructuring data.

use std::path::Path;

use ferrostat_core::dta::DtaWriter;
use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_tests::*;

fn write_dta(path: &Path, ds: &DataSet) {
    let mut f = std::fs::File::create(path).unwrap();
    DtaWriter::write(&mut f, ds).unwrap();
}

fn small(vars: &[(&str, &[f64])]) -> DataSet {
    let mut ds = DataSet::new();
    for (name, data) in vars {
        let mut v = Variable::new(*name, VarType::Double);
        v.data = data.iter().map(|&x| Value::Double(x)).collect();
        ds.add_variable(v).unwrap();
    }
    ds.n_obs = vars[0].1.len();
    ds
}

// ---- generate / replace -----------------------------------------------------

#[test]
fn generate_creates_new_variable() {
    let mut s = Session::with_fixture();
    s.run("generate w = x + 1");
    assert_eq!(s.nums("w")[0], 2.0);
    assert_eq!(s.varnames().last().unwrap(), "w");
}

#[test]
fn generate_refuses_existing_name() {
    let mut s = Session::with_fixture();
    s.fails("generate x = 1");
}

#[test]
fn generate_with_unknown_variable_fails() {
    let mut s = Session::with_fixture();
    s.fails("generate w = nosuchvar + 1");
}

#[test]
fn generate_with_if() {
    let mut s = Session::with_fixture();
    s.run("generate double w = y if group == 2");
    let w = s.nums("w");
    assert!(w[0].is_nan());
    assert_eq!(w[4], Y[4]);
}

#[test]
fn command_abbreviations() {
    let mut s = Session::with_fixture();
    s.run("gen w = 1");
    s.run("su x");
    s.run("reg y x");
    s.run("tab group");
}

#[test]
fn generate_storage_types() {
    let mut s = Session::with_fixture();
    s.run("generate byte b = 1");
    s.run("generate str10 t = \"a\"");
    assert_eq!(s.ds().get_var("b").unwrap().var_type, VarType::Byte);
    assert_eq!(s.ds().get_var("t").unwrap().var_type, VarType::Str(10));
}

#[test]
fn replace_all_values() {
    let mut s = Session::with_fixture();
    s.run("replace y = 0");
    assert!(s.nums("y").iter().all(|&v| v == 0.0));
}

#[test]
fn replace_with_if() {
    let mut s = Session::with_fixture();
    s.run("replace y = -1 if x > 10");
    let y = s.nums("y");
    assert_eq!(y[9], Y[9]);
    assert_eq!(&y[10..], [-1.0, -1.0]);
}

#[test]
fn replace_unknown_variable_fails() {
    let mut s = Session::with_fixture();
    s.fails("replace nosuch = 1");
}

#[test]
fn replace_with_in() {
    let mut s = Session::with_fixture();
    s.run("replace y = 0 in 1/2");
    assert_eq!(&s.nums("y")[..3], [0.0, 0.0, Y[2]]);
}

// ---- drop / keep ------------------------------------------------------------

#[test]
fn drop_variables() {
    let mut s = Session::with_fixture();
    s.run("drop x y");
    assert!(!s.has_var("x") && !s.has_var("y"));
    assert_eq!(s.n(), 12);
}

#[test]
fn drop_if() {
    let mut s = Session::with_fixture();
    s.run("drop if group == 1");
    assert_eq!(s.n(), 8);
    assert_eq!(s.nums("x")[0], 5.0);
}

#[test]
fn drop_in() {
    let mut s = Session::with_fixture();
    s.run("drop in 1/3");
    assert_eq!(s.n(), 9);
}

#[test]
fn drop_all() {
    let mut s = Session::with_fixture();
    s.run("drop _all");
    assert_eq!(s.n(), 0);
    assert!(s.varnames().is_empty());
}

#[test]
fn keep_variables() {
    let mut s = Session::with_fixture();
    s.run("keep id y");
    assert_eq!(s.varnames(), ["id", "y"]);
}

#[test]
fn keep_if() {
    let mut s = Session::with_fixture();
    s.run("keep if d == 1");
    assert_eq!(s.n(), 6);
    assert!(s.nums("d").iter().all(|&v| v == 1.0));
}

#[test]
fn keep_in() {
    let mut s = Session::with_fixture();
    s.run("keep in 3/5");
    assert_eq!(s.nums("x"), [3.0, 4.0, 5.0]);
}

#[test]
fn drop_if_missing_greater() {
    let mut s = Session::with_missing_fixture();
    s.run("drop if y > 20");
    assert_eq!(s.n(), 10); // 22.2 and the missing y are both dropped
}

// ---- sort -------------------------------------------------------------------

#[test]
fn sort_ascending() {
    let mut s = Session::with_fixture();
    s.run("sort z");
    let z = s.nums("z");
    assert!(z.windows(2).all(|w| w[0] <= w[1]));
}

#[test]
fn sort_multiple_keys() {
    let mut s = Session::with_fixture();
    s.run("sort d z");
    let d = s.nums("d");
    let z = s.nums("z");
    for i in 1..12 {
        assert!(d[i - 1] < d[i] || (d[i - 1] == d[i] && z[i - 1] <= z[i]));
    }
}

#[test]
fn sort_strings() {
    let mut s = Session::with_fixture();
    s.run("replace x = -x");
    s.run("sort x");
    assert_eq!(s.strs("name")[0], "lynx");
    s.run("sort name");
    assert_eq!(s.strs("name")[0], "ant");
}

#[test]
fn sort_puts_missing_last() {
    let mut s = Session::with_missing_fixture();
    s.run("sort y");
    assert!(s.nums("y")[11].is_nan());
}

#[test]
fn gsort_descending() {
    let mut s = Session::with_fixture();
    s.run("gsort -y");
    assert_eq!(s.nums("y")[0], 23.8);
}

// ---- renaming, ordering, labelling ----------------------------------------------

#[test]
fn rename_variable() {
    let mut s = Session::with_fixture();
    s.run("rename y outcome");
    assert!(s.has_var("outcome") && !s.has_var("y"));
}

#[test]
fn order_variables() {
    let mut s = Session::with_fixture();
    s.run("order y x");
    assert_eq!(&s.varnames()[..2], ["y", "x"]);
}

#[test]
fn label_variable_and_data() {
    let mut s = Session::with_fixture();
    s.run("label variable y \"Outcome\"");
    s.run("label data \"Relabelled\"");
    assert_eq!(s.ds().get_var("y").unwrap().label, "Outcome");
    assert_eq!(s.ds().datalabel, "Relabelled");
}

#[test]
fn label_define_and_values() {
    let mut s = Session::with_fixture();
    s.run("label define grp 1 \"north\" 2 \"south\" 3 \"west\"");
    s.run("label values group grp");
    assert_eq!(s.ds().get_var("group").unwrap().value_label_name, "grp");
    let out = s.run("tabulate group");
    assert!(out.contains("south"), "{out}");
}

#[test]
fn encode_and_decode() {
    let mut s = Session::with_fixture();
    s.run("encode name, generate(name_n)");
    assert_eq!(s.nums("name_n"), X); // names are already alphabetical
    s.run("decode name_n, generate(name_s)");
    assert_eq!(s.strs("name_s"), s.strs("name"));
}

#[test]
fn destring_and_tostring() {
    let mut s = Session::with_fixture();
    s.run("tostring x, generate(xs)");
    assert_eq!(s.strs("xs")[0], "1");
    s.run("destring xs, generate(xn)");
    assert_eq!(s.nums("xn"), X);
}

#[test]
fn recode_values() {
    let mut s = Session::with_fixture();
    s.run("recode group (1 2 = 0) (3 = 1), generate(g3)");
    assert_eq!(s.nums("g3"), [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.]);
}

#[test]
fn format_command() {
    let mut s = Session::with_fixture();
    s.run("format y %9.2f");
    assert_eq!(s.ds().get_var("y").unwrap().format, "%9.2f");
}

#[test]
fn compress_demotes_types() {
    let mut s = Session::with_fixture();
    s.run("compress");
    assert_eq!(s.ds().get_var("x").unwrap().var_type, VarType::Byte);
}

// ---- creating data ----------------------------------------------------------

#[test]
fn set_obs() {
    let mut s = Session::new();
    s.run("set obs 5");
    s.run("generate i = 1");
    assert_eq!(s.n(), 5);
}

#[test]
fn input_block() {
    let mut s = Session::new();
    s.script("input a b\n1 2\n3 4\nend");
    assert_eq!(s.nums("a"), [1.0, 3.0]);
    assert_eq!(s.nums("b"), [2.0, 4.0]);
}

#[test]
fn preserve_restore() {
    let mut s = Session::with_fixture();
    s.run("preserve");
    s.run("drop if group == 1");
    s.run("restore");
    assert_eq!(s.n(), 12);
}

#[test]
fn clear_empties_memory() {
    let mut s = Session::with_fixture();
    s.run("clear");
    assert_eq!(s.n(), 0);
    assert!(s.varnames().is_empty());
}

// ---- egen ---------------------------------------------------------------------

#[test]
fn egen_row_functions() {
    let mut s = Session::with_fixture();
    s.run("egen rm = rowmean(x z)");
    s.run("egen rt = rowtotal(x z)");
    s.run("egen rx = rowmax(x z)");
    s.run("egen rn = rowmin(x z)");
    for i in 0..12 {
        assert_close(s.nums("rm")[i], (X[i] + Z[i]) / 2.0, 1e-12);
        assert_close(s.nums("rt")[i], X[i] + Z[i], 1e-12);
        assert_eq!(s.nums("rx")[i], X[i].max(Z[i]));
        assert_eq!(s.nums("rn")[i], X[i].min(Z[i]));
    }
}

#[test]
fn egen_rowmiss_rownonmiss() {
    let mut s = Session::with_missing_fixture();
    s.run("egen m = rowmiss(y z)");
    s.run("egen nm = rownonmiss(y z)");
    assert_eq!(s.nums("m")[0], 1.0);
    assert_eq!(s.nums("m")[5], 0.0);
    assert_eq!(s.nums("m")[11], 1.0);
    assert_eq!(s.nums("nm")[5], 2.0);
}

#[test]
fn egen_rowtotal_treats_missing_as_zero() {
    let mut s = Session::with_missing_fixture();
    s.run("egen t = rowtotal(y z)");
    assert_eq!(s.nums("t")[11], Z[11]);
}

#[test]
fn egen_column_statistics() {
    let mut s = Session::with_fixture();
    s.run("egen double m = mean(y)");
    s.run("egen double sd = sd(y)");
    s.run("egen double mx = max(y)");
    s.run("egen double mn = min(y)");
    s.run("egen double t = total(y)");
    s.run("egen c = count(y)");
    assert_close(s.nums("m")[5], reference::Y_MEAN, 1e-12);
    assert_close(s.nums("sd")[5], reference::Y_SD, 1e-12);
    assert_eq!(s.nums("mx")[0], 23.8);
    assert_eq!(s.nums("mn")[0], 2.1);
    assert_close(s.nums("t")[0], Y.iter().sum(), 1e-12);
    assert_eq!(s.nums("c")[0], 12.0);
}

#[test]
fn egen_by_group() {
    let mut s = Session::with_fixture();
    s.run("bysort group: egen double gm = mean(y)");
    assert_all_close(&[s.nums("gm")[0], s.nums("gm")[4], s.nums("gm")[8]], &reference::GROUP_MEAN_Y, 1e-12);
    s.run("egen double gm2 = mean(y), by(group)");
    assert_eq!(s.nums("gm"), s.nums("gm2"));
}

#[test]
fn egen_group_numbers_categories() {
    let mut s = Session::with_fixture();
    s.run("egen g = group(group d)");
    let g = s.nums("g");
    // (1,0),(1,1),(2,0),(2,1),(3,0),(3,1) => 6 distinct groups, numbered 1..6 in sort order
    let mut distinct = g.clone();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    assert_eq!(distinct, [1., 2., 3., 4., 5., 6.]);
    assert_eq!(g[0], 1.0); // group 1, d 0
    assert_eq!(g[2], 2.0); // group 1, d 1
}

#[test]
fn egen_rank_with_ties_averaged() {
    let mut s = Session::with_fixture();
    s.run("egen r = rank(z)");
    // z = 3 1 4 1 5 9 2 6 5 3 5 8 -> 1s share ranks 1.5, 5s share rank 8
    let r = s.nums("r");
    assert_eq!(r[1], 1.5);
    assert_eq!(r[3], 1.5);
    assert_eq!(r[4], 8.0);
    assert_eq!(r[5], 12.0);
}

#[test]
fn egen_seq() {
    let mut s = Session::with_fixture();
    s.run("egen q = seq()");
    assert_eq!(s.nums("q"), X);
}

#[test]
fn egen_more_functions() {
    let mut s = Session::with_fixture();
    s.run("egen double med = median(y)");
    assert_close(s.nums("med")[0], reference::Y_P50, 1e-12);
    s.run("egen double p75 = pctile(y), p(75)");
    assert_close(s.nums("p75")[0], reference::Y_P75, 1e-12);
    s.run("egen t = tag(group)");
    assert_eq!(s.nums("t").iter().sum::<f64>(), 3.0);
    s.run("egen double sy = std(y)");
    assert_close(s.nums("sy")[0], (Y[0] - reference::Y_MEAN) / reference::Y_SD, 1e-12);
    s.run("egen c = cut(x), at(0 4 8 13)");
    assert_eq!(s.nums("c")[4], 4.0);
}

#[test]
fn egen_unknown_function_fails() {
    let mut s = Session::with_fixture();
    s.fails("egen q = nosuchfn(y)");
}

// ---- by-group processing ------------------------------------------------------

#[test]
fn by_group_running_counter() {
    let mut s = Session::with_fixture();
    s.run("bysort group: generate k = _n");
    s.run("bysort group: generate kk = _N");
    assert_eq!(&s.nums("k")[..5], [1., 2., 3., 4., 1.]);
    assert!(s.nums("kk").iter().all(|&v| v == 4.0));
}

#[test]
fn by_prefix_summarize() {
    let mut s = Session::with_fixture();
    let out = s.run("by group, sort: summarize y");
    assert_eq!(out.matches("Obs").count(), 3, "{out}");
}

// ---- collapse / contract ------------------------------------------------------

#[test]
fn collapse_mean_by_group() {
    let mut s = Session::with_fixture();
    s.run("collapse (mean) y, by(group)");
    assert_eq!(s.n(), 3);
    assert_all_close(&s.nums("y"), &reference::GROUP_MEAN_Y, 1e-12);
}

#[test]
fn collapse_default_statistic_is_mean() {
    let mut s = Session::with_fixture();
    s.run("collapse y x");
    assert_eq!(s.n(), 1);
    assert_close(s.nums("y")[0], reference::Y_MEAN, 1e-12);
    assert_close(s.nums("x")[0], reference::X_MEAN, 1e-12);
}

#[test]
fn collapse_by_without_stat_is_mean() {
    let mut s = Session::with_fixture();
    s.run("collapse y, by(group)");
    assert_eq!(s.n(), 3);
    assert_eq!(s.nums("group"), [1., 2., 3.]);
    assert_all_close(&s.nums("y"), &reference::GROUP_MEAN_Y, 1e-12);
}

#[test]
fn collapse_multiple_statistics_with_rename() {
    let mut s = Session::with_fixture();
    s.run("collapse (sum) ys=y (sd) ysd=y (count) n=y, by(group)");
    assert_all_close(&s.nums("ys"), &reference::GROUP_SUM_Y, 1e-12);
    assert_all_close(&s.nums("ysd"), &reference::GROUP_SD_Y, 1e-12);
    assert_eq!(s.nums("n"), [4., 4., 4.]);
}

#[test]
fn contract_frequencies() {
    let mut s = Session::with_fixture();
    s.run("contract group d");
    assert_eq!(s.n(), 6);
    assert!(s.has_var("_freq"));
    assert_eq!(s.nums("_freq").iter().sum::<f64>(), 12.0);
}

// ---- expand / fillin --------------------------------------------------------

#[test]
fn expand_constant() {
    let mut s = Session::with_fixture();
    s.run("expand 2");
    assert_eq!(s.n(), 24);
}

#[test]
fn expand_by_variable() {
    let mut s = Session::with_fixture();
    s.run("expand cnt");
    // obs with cnt == 0 are kept once (Stata keeps n<1 as 1 observation)
    let expected: f64 = CNT.iter().map(|&c| c.max(1.0)).sum();
    assert_eq!(s.n() as f64, expected);
}

#[test]
fn fillin_rectangularizes() {
    let mut s = Session::with_fixture();
    s.run("keep group z");
    s.run("fillin group z");
    // 3 groups x 8 distinct z values = 24 cells; 10 cells exist (two of them
    // twice), so 14 rows are added and the 12 originals are kept.
    let added = s.nums("_fillin").iter().filter(|&&v| v == 1.0).count();
    assert_eq!(added, 14);
    assert_eq!(s.n(), 26);
}

// ---- reshape ----------------------------------------------------------------

#[test]
fn reshape_wide_to_long() {
    let mut s = Session::new();
    s.load(small(&[("id", &[1., 2.]), ("inc1", &[10., 20.]), ("inc2", &[11., 21.])]));
    s.run("reshape long inc, i(id) j(year)");
    assert_eq!(s.n(), 4);
    assert!(s.has_var("inc"));
}

#[test]
fn reshape_long_numeric_j() {
    let mut s = Session::new();
    s.load(small(&[("id", &[1., 2.]), ("inc1", &[10., 20.]), ("inc2", &[11., 21.])]));
    s.run("reshape long inc, i(id) j(year)");
    assert_eq!(s.nums("year"), [1., 2., 1., 2.]);
    assert_eq!(s.nums("inc"), [10., 11., 20., 21.]);
}

#[test]
fn reshape_long_to_wide() {
    let mut s = Session::new();
    s.load(small(&[("id", &[1., 1., 2., 2.]), ("year", &[1., 2., 1., 2.]), ("inc", &[10., 11., 20., 21.])]));
    s.run("reshape wide inc, i(id) j(year)");
    assert_eq!(s.n(), 2);
    assert!(s.has_var("inc1") && s.has_var("inc2"));
    assert_eq!(s.nums("inc2"), [11., 21.]);
}

// ---- append / merge ---------------------------------------------------------

#[test]
fn append_using() {
    let dir = temp_dir("append");
    let other = dir.join("other.dta");
    write_dta(&other, &small(&[("x", &[100., 200.]), ("extra", &[1., 2.])]));
    let mut s = Session::with_fixture();
    s.run(&format!("append using {}", other.display()));
    assert_eq!(s.n(), 14);
    assert_eq!(s.nums("x")[12..], [100., 200.]);
    assert!(s.nums("y")[13].is_nan());
    assert!(s.nums("extra")[0].is_nan());
}

#[test]
fn merge_one_to_one() {
    let dir = temp_dir("merge11");
    let other = dir.join("other.dta");
    write_dta(&other, &small(&[("id", &[1., 2., 99.]), ("w", &[0.5, 0.25, 9.0])]));
    let mut s = Session::with_fixture();
    s.run(&format!("merge 1:1 id using {}", other.display()));
    assert_eq!(s.n(), 13);
    let m = s.nums("_merge");
    assert_eq!(m.iter().filter(|&&v| v == 3.0).count(), 2); // matched
    assert_eq!(m.iter().filter(|&&v| v == 1.0).count(), 10); // master only
    assert_eq!(m.iter().filter(|&&v| v == 2.0).count(), 1); // using only
}

#[test]
fn merge_many_to_one() {
    let dir = temp_dir("mergem1");
    let other = dir.join("groups.dta");
    write_dta(&other, &small(&[("group", &[1., 2., 3.]), ("gname", &[10., 20., 30.])]));
    let mut s = Session::with_fixture();
    s.run(&format!("merge m:1 group using {}", other.display()));
    assert_eq!(s.n(), 12);
    assert_eq!(s.nums("gname")[5], 20.0);
}

#[test]
fn merge_options() {
    let dir = temp_dir("mergeopts");
    let other = dir.join("other.dta");
    write_dta(&other, &small(&[("id", &[1., 2., 99.]), ("w", &[0.5, 0.25, 9.0])]));
    let mut s = Session::with_fixture();
    s.run(&format!("merge 1:1 id using {}, keep(match) nogenerate", other.display()));
    assert_eq!(s.n(), 2);
    assert!(!s.has_var("_merge"));
}

#[test]
fn joinby_and_cross() {
    let dir = temp_dir("cross");
    let other = dir.join("other.dta");
    write_dta(&other, &small(&[("k", &[1., 2.])]));
    let mut s = Session::with_fixture();
    s.run(&format!("cross using {}", other.display()));
    assert_eq!(s.n(), 24);
}

// ---- duplicates, misc ---------------------------------------------------------

#[test]
fn duplicates_drop() {
    let mut s = Session::with_fixture();
    s.run("keep group");
    s.run("duplicates drop");
    assert_eq!(s.n(), 3);
}

#[test]
fn xpose() {
    let mut s = Session::with_fixture();
    s.run("keep x y");
    s.run("xpose, clear");
    assert_eq!(s.n(), 2);
}

#[test]
fn sample_fraction() {
    let mut s = Session::with_fixture();
    s.run("set seed 1");
    s.run("sample 50");
    assert_eq!(s.n(), 6);
}

#[test]
fn rename_case_options() {
    // `rename *, lower` (test-andrea/Spatial.do)
    let mut s = Session::with_fixture();
    s.run("rename x X_Big");
    s.run("rename *, upper");
    assert!(s.has_var("X_BIG") && s.has_var("ID"));
    s.run("rename *, lower");
    assert!(s.has_var("x_big") && s.has_var("id"));
    s.run("rename x_big, proper");
    assert!(s.has_var("X_big"));
}
