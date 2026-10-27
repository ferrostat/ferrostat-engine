//! twoway plot lists with a shared if/in, plot-type abbreviations, fill
//! colours and legend labels, pie charts and regression diagnostic plots.

use ferrostat_tests::*;

/// Save the current graph and return its text form, one entry per line.
fn saved(s: &mut Session, test: &str) -> Vec<String> {
    let path = temp_dir(test).join("g.gph");
    s.run(&format!("graph save \"{}\", replace", path.display()));
    std::fs::read_to_string(&path).unwrap().lines().map(str::to_string).collect()
}

/// Points of each layer of the saved graph.
fn layer_points(lines: &[String]) -> Vec<Vec<(f64, f64)>> {
    let mut out: Vec<Vec<(f64, f64)>> = Vec::new();
    for l in lines {
        let f: Vec<&str> = l.split('\t').collect();
        match f[0] {
            "layer" => out.push(Vec::new()),
            "p" => out.last_mut().unwrap().push((f[1].parse().unwrap(), f[2].parse().unwrap())),
            _ => {}
        }
    }
    out
}

fn layers(lines: &[String]) -> Vec<Vec<String>> {
    lines.iter().filter(|l| l.starts_with("layer\t")).map(|l| l.split('\t').map(str::to_string).collect()).collect()
}

fn pies(lines: &[String]) -> Vec<(f64, String, String)> {
    lines.iter().filter(|l| l.starts_with("pie\t")).map(|l| {
        let f: Vec<&str> = l.split('\t').collect();
        (f[1].parse().unwrap(), f[3].to_string(), f[4].to_string())
    }).collect()
}

#[test]
fn trailing_if_applies_to_every_plot() {
    let mut s = Session::with_fixture();
    s.run("twoway (scatter y x) (line y x) if group == 1");
    let pts = layer_points(&saved(&mut s, "t05_if"));
    assert_eq!(pts.len(), 2);
    assert_eq!(pts[0].len(), 4);
    assert_eq!(pts[1].len(), 4);
    // combined with a plot's own condition
    s.run("twoway (scatter y x if d == 1) (scatter y x) if group <= 2 in 1/6, legend(off)");
    let pts = layer_points(&saved(&mut s, "t05_if2"));
    assert_eq!(pts[0].len(), 2);
    assert_eq!(pts[1].len(), 6);
    // the || form with a final if segment
    s.run("twoway scatter y x || line y x || if group == 3");
    let pts = layer_points(&saved(&mut s, "t05_if3"));
    assert!(pts.iter().all(|p| p.len() == 4));
}

#[test]
fn trailing_if_with_by_and_lfit() {
    let mut s = Session::with_fixture();
    s.run("gen byte touse = id > 2");
    s.run("twoway (scatter y x) (lfit y x) if touse, by(d)");
    let dir = temp_dir("t05_by");
    let png = dir.join("by.png");
    s.run(&format!("graph export \"{}\", replace", png.display()));
    assert_eq!(&std::fs::read(&png).unwrap()[1..4], b"PNG");
}

#[test]
fn plot_type_abbreviations_and_fill_colours() {
    let mut s = Session::with_fixture();
    s.run("twoway (hist x if d==0, color(blue%25) w(2) start(0)) (hist x if d==1, fcolor(none) lcolor(red) w(2) start(0)), legend(lab(1 \"Old\") lab(2 \"New\"))");
    let g = saved(&mut s, "t05_hist");
    let ls = layers(&g);
    assert_eq!(ls.len(), 2);
    assert_eq!(ls[0][3], "Old");
    assert_eq!(ls[1][3], "New");
    assert_eq!(ls[0][6], "0.25"); // opacity
    assert_eq!(ls[0][10], "0 0 255");
    assert_eq!(ls[1][11], "true"); // outline only
    assert_eq!(ls[1][10], "255 0 0");
    assert_eq!(ls[0][2], "2");
    s.run("twoway (sc y x) (conn y x) (li y x)");
    assert_eq!(layers(&saved(&mut s, "t05_abbrev")).len(), 3);
}

#[test]
fn legend_inside_plot_options_is_merged() {
    let mut s = Session::with_fixture();
    s.run("twoway (hist x if d==0, col(blue%25) w(1) legend(lab(1 Old))) (hist x if d==1, col(red%25) w(1) legend(lab(2 New)))");
    let ls = layers(&saved(&mut s, "t05_leg"));
    assert_eq!(ls[0][3], "Old");
    assert_eq!(ls[1][3], "New");
}

#[test]
fn pie_variants() {
    let mut s = Session::with_fixture();
    // counts per category
    s.run("graph pie, over(group) plabel(_all sum)");
    let p = pies(&saved(&mut s, "t05_pie1"));
    assert_eq!(p.iter().map(|x| x.0).collect::<Vec<_>>(), [4.0, 4.0, 4.0]);
    assert_eq!(p[0].2, "4");
    // sums of a variable per category, percent labels
    s.run("graph pie z, over(d) plabel(_all percent) pie(2, explode) sort angle0(0) title(\"Z\")");
    let p = pies(&saved(&mut s, "t05_pie2"));
    assert_eq!(p.len(), 2);
    assert!(p[0].0 <= p[1].0);
    let total: f64 = Z.iter().sum();
    let pct = (p[0].0 / total * 1000.0).round() / 10.0;
    assert_eq!(p[0].2, format!("{pct}%"));
    // several variables: slices are their sums, keyed by variable label
    s.run("graph pie cnt z, plabel(_all name)");
    let p = pies(&saved(&mut s, "t05_pie3"));
    assert_eq!(p[0].0, CNT.iter().sum::<f64>());
    assert_eq!(p[1].0, total);
    s.fails("graph pie");
}

#[test]
fn residual_versus_fitted() {
    let mut s = Session::with_fixture();
    s.fails("rvfplot");
    s.run("regress y x z");
    s.run("rvfplot, yline(0)");
    let g = saved(&mut s, "t05_rvf");
    assert!(g.contains(&"ytitle\tResiduals".to_string()));
    assert!(g.contains(&"ylines\t0".to_string()));
    let pts = &layer_points(&g)[0];
    assert_eq!(pts.len(), 12);
    // residuals sum to zero
    assert!(pts.iter().map(|p| p.1).sum::<f64>().abs() < 1e-8);
    s.run("predict double yh");
    assert_all_close(&pts.iter().map(|p| p.0).collect::<Vec<_>>(), &s.nums("yh"), 1e-9);
    s.run("rvpplot z");
    assert_eq!(layer_points(&saved(&mut s, "t05_rvp"))[0].len(), 12);
    s.run("lvr2plot");
    let pts = &layer_points(&saved(&mut s, "t05_lvr"))[0];
    // normalized squared residuals sum to one; leverages sum to k
    assert_close(pts.iter().map(|p| p.0).sum::<f64>(), 1.0, 1e-9);
    assert_close(pts.iter().map(|p| p.1).sum::<f64>(), 3.0, 1e-8);
}

#[test]
fn added_variable_plot_slope_is_coefficient() {
    let mut s = Session::with_fixture();
    s.run("regress y x z");
    let b = s.scalar("_b[z]");
    s.run("avplot z");
    let pts = layer_points(&saved(&mut s, "t05_av"));
    // least-squares slope through the residual cloud equals _b[z]
    let (sxy, sxx): (f64, f64) = pts[0].iter().fold((0.0, 0.0), |a, p| (a.0 + p.0 * p.1, a.1 + p.0 * p.0));
    assert_close(sxy / sxx, b, 1e-8);
    let line = &pts[1];
    assert_close((line[1].1 - line[0].1) / (line[1].0 - line[0].0), b, 1e-9);
    s.run("avplots");
    s.fails("avplot d");
}

#[test]
fn nested_over_labels_and_eps_export() {
    let mut s = Session::with_fixture();
    s.run("graph bar y, over(d) over(group)");
    let g = saved(&mut s, "t05_nested");
    let xl: Vec<&String> = g.iter().filter(|l| l.starts_with("xlabel\t")).collect();
    // six inner labels plus one outer label per group
    assert_eq!(xl.len(), 9);
    assert!(g.contains(&"xlabel\t3\t\\n2".to_string()), "{xl:?}");
    let path = temp_dir("t05_eps").join("g.eps");
    let out = s.run(&format!("graph export \"{}\", replace", path.display()));
    assert!(out.contains("EPS"), "{out}");
    assert!(std::fs::read_to_string(&path).unwrap().starts_with("%!PS-Adobe-3.0 EPSF-3.0"));
}
