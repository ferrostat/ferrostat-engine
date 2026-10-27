//! The matrix language: blocks, one-liners, operators, functions, display,
//! the data/macro interface and the spreadsheet class.

use ferrostat_tests::*;

fn mata(s: &mut Session, code: &str) -> String {
    s.run(&format!("mata\n{code}\nend"))
}

#[test]
fn matrix_literals_display_and_operators() {
    let mut s = Session::new();
    let out = mata(&mut s, "A = (1,2\\3,4)\nA");
    assert!(out.contains("       1   2\n    +---------+\n  1 |  1   2  |\n  2 |  3   4  |\n    +---------+"), "{out}");
    let out = mata(&mut s, "B = A'\nB*A");
    assert!(out.contains("  1 |  10   14  |") && out.contains("  2 |  14   20  |"), "{out}");
    assert!(mata(&mut s, "A:*A").contains("  2 |   9   16  |"));
    assert_eq!(mata(&mut s, "2+2").trim(), "4");
    assert_eq!(mata(&mut s, "sqrt(2)").trim(), "1.414213562");
    assert_eq!(mata(&mut s, "1/3").trim(), ".3333333333");
    assert_eq!(mata(&mut s, "-2^2").trim(), "-4");
    assert_eq!(mata(&mut s, "sum(1..10)").trim(), "55");
    assert_eq!(mata(&mut s, "rows(1::7), cols(1..3)").contains("  1 |  7   3  |"), true);
    // joins, subscripts, ranges
    let out = mata(&mut s, "C = (1::4),(11::14)\nC[|2,1\\3,2|]");
    assert!(out.contains("  1 |   2   12  |") && out.contains("  2 |   3   13  |"), "{out}");
    assert_eq!(mata(&mut s, "C[4,2]").trim(), "14");
    assert!(mata(&mut s, "C[(1\\4),.]").contains("  2 |   4   14  |"));
    // X'X means X'*X
    assert!(mata(&mut s, "X = (1,2\\3,4)\nX'X").contains("  1 |  10   14  |"));
    // strings
    let out = mata(&mut s, "S = (\"Hello\", \"How\" \\ \"Are\", \"You?\")\nS");
    assert!(out.contains("  1 |  Hello     How  |"), "{out}");
    assert_eq!(mata(&mut s, "\"ab\" + \"cd\"").trim(), "abcd");
    assert_eq!(mata(&mut s, "strupper(\"ab\") + substr(\"hello\", 2, 3)").trim(), "ABell");
    assert_eq!(mata(&mut s, "invtokens(tokens(\"a b  c\"), \",\")").trim(), "a,b,c");
    // conformability errors are reported and execution continues
    let out = mata(&mut s, "(1,2) + (1\\2)\n5*5");
    assert!(out.contains("3200") && out.contains("conformability error") && out.contains("25"), "{out}");
    // wide matrices are split into column segments
    let out = mata(&mut s, "(101..120)\\(111..130)");
    assert!(out.contains("        13    14") && out.contains("  |"), "{out}");
}

#[test]
fn colon_operators_select_and_library() {
    let mut s = Session::new();
    mata(&mut s, "x = (11,2,41,70,5,60,100,22)");
    assert!(mata(&mut s, "select(x, x:>10)").contains("  1 |   11    41    70    60   100    22  |"));
    assert!(mata(&mut s, "selectindex(x:<10)").contains("  1 |  2   5  |"));
    assert!(mata(&mut s, "sort((3\\1\\2), 1)'").contains("  1 |  1   2   3  |"));
    assert_eq!(mata(&mut s, "max(x) - min(x)").trim(), "98");
    assert!(mata(&mut s, "colsum((1,2\\3,4))").contains("  1 |  4   6  |"));
    assert!(mata(&mut s, "mean((1,2\\3,4))").contains("  1 |  2   3  |"));
    assert!(mata(&mut s, "invsym((2,0\\0,4))").contains("  1 |   .5     0  |"));
    assert_eq!(mata(&mut s, "det((1,2\\3,4))").trim(), "-2");
    assert!(mata(&mut s, "J(2,3,0)").contains("  2 |  0   0   0  |"));
    assert!(mata(&mut s, "I(2)*3").contains("  2 |  0   3  |"));
    assert!(mata(&mut s, "(1\\2) :* (1,2)").contains("  2 |  2   4  |"));
    assert_eq!(mata(&mut s, "sprintf(\"%5.2f|%s\", 3.14159, \"x\")").trim(), "3.14|x");
    assert_eq!(mata(&mut s, "printf(\"v=%g\\n\", 7)").trim(), "v=7");
    assert_eq!(mata(&mut s, "strofreal(12.5) + \"!\"").trim(), "12.5!");
    assert_eq!(mata(&mut s, "missing((1,.,3))").trim(), "1");
    // host scalar functions are available elementwise
    assert_eq!(mata(&mut s, "normal(0)").trim(), ".5");
    assert_eq!(mata(&mut s, "cross((1\\2), (3\\4))").trim(), "11");
}

#[test]
fn functions_loops_and_declarations() {
    let mut s = Session::new();
    let out = mata(
        &mut s,
        "real scalar function ownfunc(real vector x) {
    real vector y
    real scalar c, p
    y = select(x, x:>10)
    c = length(y)
    p = 1
    for (i=3; i<=c; i=i+3) {
        p = p * y[i]
    }
    return(p)
}
ownfunc((11,2,41,70,5,60,100,22))
ownfunc((10,20\\30,40))",
    );
    assert!(out.contains("1540"), "{out}");
    assert!(out.contains("ownfunc():  3201  vector required"), "{out}");
    // functions persist across blocks; while, do-while, break, continue, ++, +=
    let out = mata(
        &mut s,
        "function f(n) {
    k = 0
    t = 0
    while (1) {
        k++
        if (k > n) break
        if (mod(k, 2) == 0) continue
        t += k
    }
    do {
        t = t - 1
    } while (t > 20)
    return(t)
}",
    );
    assert!(out.trim().is_empty(), "{out}");
    assert_eq!(mata(&mut s, "f(9)").trim(), "20");
    // arguments are passed by reference
    mata(&mut s, "void setfirst(x) {\n x[1] = 99\n}\nv = (1,2,3)\nsetfirst(v)");
    assert!(mata(&mut s, "v").contains("  1 |  99    2    3  |"));
    // type checks
    let out = mata(&mut s, "real matrix function g(real matrix a) {\n return(a)\n}\ng(\"x\")");
    assert!(out.contains("3251") && out.contains("nonreal found where real required"), "{out}");
    // pointers
    assert_eq!(mata(&mut s, "p = &v\n(*p)[2]").trim(), "2");
    // exit() stops with the given return code
    let out = mata(&mut s, "function h() {\n display(\"stop\")\n exit(1)\n}\nh()\n3");
    assert!(out.contains("stop") && out.contains("r(1);") && out.contains("3"), "{out}");
    // mata describe / drop
    let out = s.run("mata: mata describe");
    assert!(out.contains("ownfunc()") && out.contains("v[1,3]"), "{out}");
    s.run("mata drop v");
    assert!(mata(&mut s, "v").contains("v not found"));
    s.run("mata drop *()");
    assert!(mata(&mut s, "f(1)").contains("f() not found"));
    s.run("clear mata");
    assert!(mata(&mut s, "p").contains("not found"));
}

#[test]
fn colon_block_and_one_liners_stop_on_error() {
    let mut s = Session::new();
    assert!(s.try_run("mata:\nsome mistake\nend").is_err());
    let e = s.try_run("mata: (1,2)+(1\\2)").unwrap_err();
    assert!(e.contains("r(3200);"), "{e}");
    s.run("mata: a = 5");
    assert_eq!(s.run("mata: a*2").trim(), "10");
    // values persist across one-liners and blocks
    assert_eq!(mata(&mut s, "a").trim(), "5");
}

#[test]
fn data_interface() {
    let mut s = Session::with_fixture();
    assert_eq!(s.run("mata: st_nobs()").trim(), "12");
    assert_eq!(s.run("mata: st_nvar()").trim(), "8");
    let total: f64 = s.nums("x").iter().sum();
    let out = s.run("mata: colsum(st_data(., \"x\"))");
    assert_close(out.trim().parse().unwrap(), total, 1e-9);
    // selection variable: observations where d != 0
    let nd = s.nums("d").iter().filter(|v| **v != 0.0).count();
    assert_eq!(s.run("mata: rows(st_data(., (\"x\", \"y\"), \"d\"))").trim(), nd.to_string());
    let names = s.strs("name");
    assert_eq!(s.run("mata: st_sdata(2, \"name\")").trim(), names[1]);
    // views write back into the data
    mata(&mut s, "st_view(V=., ., \"y\")\nV[1] = 1234");
    assert_eq!(s.nums("y")[0], 1234.0);
    // new variables, stores, observations
    mata(&mut s, "idx = st_addvar(\"double\", \"w\")\nst_store(., \"w\", (1::12):*2)");
    assert_eq!(s.nums("w")[11], 24.0);
    mata(&mut s, "st_addvar(\"str5\", \"lab\")\nst_sstore(1, \"lab\", \"hello\")");
    assert_eq!(s.strs("lab")[0], "hello");
    mata(&mut s, "st_addobs(2)");
    assert_eq!(s.n(), 14);
    mata(&mut s, "st_dropvar(\"w\")");
    assert!(!s.has_var("w"));
    assert_eq!(s.run("mata: st_varindex(\"x\")").trim(), "4");
    assert_eq!(s.run("mata: st_varname(2)").trim(), "name");
    // macros, scalars, matrices
    mata(&mut s, "st_local(\"lm\", \"abc\")\nst_global(\"gm\", strofreal(42))\nst_numscalar(\"sc\", 3.5)\nst_matrix(\"M\", (1,2\\3,4))");
    assert_eq!(s.run("display \"`lm'\"").trim(), "abc");
    assert_eq!(s.scalar("$gm"), 42.0);
    assert_eq!(s.scalar("sc"), 3.5);
    assert_eq!(s.scalar("M[2,1]"), 3.0);
    assert_eq!(s.run("mata: st_global(\"gm\")").trim(), "42");
    assert!(s.run("mata: st_matrix(\"M\")").contains("  2 |  3   4  |"));
    s.run("mata: st_matrixrowstripe(\"M\", (\"\", \"a\" \\ \"\", \"b\"))");
    assert!(s.run("mata: st_matrixrowstripe(\"M\")").contains("a"));
    // looping over variables with one-liners
    let out = s.run("foreach v in x y {\nmata: st_view(X=.,.,\"`v'\")\nmata: rows(X)\n}");
    assert!(out.contains("14"), "{out}");
}

#[test]
fn ols_by_cross_products() {
    let mut s = Session::with_fixture();
    s.run("regress y x z");
    let bx = s.scalar("_b[x]");
    let bc = s.scalar("_b[_cons]");
    mata(&mut s, "st_view(y=., ., \"y\")\nst_view(X=., ., (\"x\", \"z\"))\nb = invsym(cross(X,1,X,1))*cross(X,1,y,0)\nb2 = invsym((X,J(rows(X),1,1))'(X,J(rows(X),1,1)))*(X,J(rows(X),1,1))'y\nst_matrix(\"B\", b')\nst_matrix(\"B2\", b2')");
    assert_close(s.scalar("B[1,1]"), bx, 1e-8);
    assert_close(s.scalar("B[1,3]"), bc, 1e-8);
    assert_close(s.scalar("B2[1,1]"), bx, 1e-8);
}

#[test]
fn spreadsheet_class() {
    let dir = temp_dir("t12_mata_xl");
    let path = dir.join("book.xlsx");
    let p = path.to_string_lossy().replace('\\', "/");
    let mut s = Session::new();
    mata(&mut s, &format!("b = xl()\nb.create_book(\"{p}\", \"first\")\nb.put_string(1, 1, (\"a\", \"b\"))\nb.put_number(2, 1, (1, 2))\nb.add_sheet(\"second\")\nb.close_book()"));
    let out = mata(&mut s, &format!("c = xl()\nc.load_book(\"{p}\")\nsh = c.get_sheets()\nst_local(\"sheets\", invtokens(sh', \",\"))\nc.set_sheet(\"first\")\nc.get_string(1, (1,2))\nc.get_number(2, \"B\")\nc.get_cell_type(3, 1)"));
    assert_eq!(s.run("display \"`sheets'\"").trim(), "first,second");
    assert!(out.contains("  1 |  a   b  |"), "{out}");
    assert!(out.contains("  2") && out.contains("blank"), "{out}");
}
