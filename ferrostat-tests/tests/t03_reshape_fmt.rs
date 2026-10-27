//! reshape settings remembered with the data, and output-formatting details
//! (float rendering, regression header widths, estimates table layout,
//! return-list order).

use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_tests::*;

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

fn wide() -> DataSet {
    small(&[("id", &[1., 2., 3.]), ("inc80", &[10., 20., 30.]), ("inc81", &[11., 21., 31.]), ("age80", &[30., 40., 50.]), ("age81", &[31., 41., 51.])])
}

#[test]
fn reshape_back_and_forth_from_memory() {
    let mut s = Session::new();
    s.load(wide());
    s.run("reshape long inc age, i(id) j(year)");
    assert_eq!(s.n(), 6);
    s.run("reshape wide");
    assert_eq!(s.n(), 3);
    assert_eq!(s.nums("inc81"), [11., 21., 31.]);
    s.run("reshape long");
    assert_eq!(s.n(), 6);
    assert_eq!(s.nums("year"), [80., 81., 80., 81., 80., 81.]);
    assert_eq!(s.nums("age"), [30., 31., 40., 41., 50., 51.]);
}

#[test]
fn reshape_summary_lists_j_values_and_xij() {
    let mut s = Session::new();
    s.load(wide());
    let out = s.run("reshape long inc age, i(id) j(year)");
    assert!(out.starts_with("(j = 80 81)"), "{out}");
    assert!(out.contains("j variable (2 values)"), "{out}");
    assert!(out.contains("inc80 inc81   ->   inc"), "{out}");
    let out = s.run("reshape wide");
    assert!(out.contains("year   ->   (dropped)"), "{out}");
    assert!(out.contains("inc   ->   inc80 inc81"), "{out}");
}

#[test]
fn reshape_query_and_clear() {
    let mut s = Session::new();
    s.load(wide());
    s.run("reshape long inc age, i(id) j(year)");
    let q = s.run("reshape");
    assert!(q.contains("data are long") && q.contains("inc age") && q.contains("year"), "{q}");
    s.run("reshape clear");
    let q = s.run("reshape");
    assert!(!q.contains("(data are long)"), "{q}");
    s.fails("reshape wide");
}

#[test]
fn reshape_settings_survive_save_and_use() {
    let dir = temp_dir("t03_reshape_save");
    let f = dir.join("r.dta");
    let mut s = Session::new();
    s.load(wide());
    s.run("reshape long inc age, i(id) j(year)");
    s.run(&format!("save \"{}\", replace", f.display()));
    s.run("clear");
    s.run(&format!("use \"{}\", clear", f.display()));
    s.run("reshape wide");
    assert_eq!(s.n(), 3);
    assert!(s.has_var("age81"));
}

#[test]
fn reshape_error_lists_duplicates() {
    let mut s = Session::new();
    s.load(small(&[("id", &[1., 1., 2.]), ("t", &[1., 1., 1.]), ("x", &[5., 6., 7.])]));
    s.fails("reshape wide x, i(id) j(t)");
    let out = s.run("reshape error");
    assert!(out.contains("(2 of 3 observations have duplicate i-j values)"), "{out}");
}

#[test]
fn list_float_rounds_to_float_precision() {
    let mut s = Session::new();
    s.run("set obs 2");
    s.run("gen float f = cond(_n == 1, 0.4, 0.40000004)");
    let out = s.run("list");
    assert!(!out.contains("40000"), "{out}");
    assert_eq!(out.matches(" .4 ").count(), 2, "{out}");
}

#[test]
fn regress_header_widths() {
    let mut s = Session::with_fixture();
    let out = s.run("regress y x z");
    let rmse = out.lines().find(|l| l.contains("Root MSE")).unwrap();
    let v = rmse.rsplit('=').next().unwrap().trim();
    assert!(v.chars().filter(|c| c.is_ascii_digit()).count() <= 5, "{rmse}");
    let f = out.lines().find(|l| l.contains("F(2, 9)")).unwrap();
    assert!(f.contains("F(2, 9)         = "), "{f}");
}

#[test]
fn estimates_table_cons_last_and_factor_groups() {
    let mut s = Session::with_fixture();
    s.run("regress y x");
    s.run("estimates store e1");
    s.run("regress y x i.group");
    s.run("estimates store e2");
    let out = s.run("estimates table e1 e2");
    let lines: Vec<&str> = out.lines().collect();
    let cons = lines.iter().position(|l| l.trim_start().starts_with("_cons")).unwrap();
    let grp = lines.iter().position(|l| l.trim() == "group |").expect(&out);
    assert!(grp < cons, "{out}");
    assert!(!out.contains("1b.group") && !out.contains("2.group"), "{out}");
    assert!(lines.iter().any(|l| l.trim_start().starts_with("2 |")), "{out}");
}

#[test]
fn summarize_return_list_order() {
    let mut s = Session::with_fixture();
    s.run("summarize x");
    let out = s.run("return list");
    let keys: Vec<&str> = out.lines().filter_map(|l| l.trim().strip_prefix("r(")).map(|l| l.split(')').next().unwrap()).collect();
    assert_eq!(keys, ["N", "sum_w", "mean", "Var", "sd", "min", "max", "sum"]);
}
