//! Shapefile import, contiguity weights, spatial lags and maps
//! (the workflow of test-andrea/DFA.do section 5 and Spatial.do task 4).

use std::path::Path;

use ferrostat_tests::*;

/// Write a shapefile of `cols` x `rows` unit squares with a NAME attribute.
fn write_grid_shapefile(base: &Path, cols: usize, rows: usize) {
    let mut recs: Vec<Vec<u8>> = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let (x, y) = (c as f64, r as f64);
            let pts = [(x, y), (x, y + 1.0), (x + 1.0, y + 1.0), (x + 1.0, y), (x, y)];
            let mut b = Vec::new();
            b.extend_from_slice(&5i32.to_le_bytes());
            for v in [x, y, x + 1.0, y + 1.0] {
                b.extend_from_slice(&v.to_le_bytes());
            }
            b.extend_from_slice(&1i32.to_le_bytes());
            b.extend_from_slice(&(pts.len() as i32).to_le_bytes());
            b.extend_from_slice(&0i32.to_le_bytes());
            for (px, py) in pts {
                b.extend_from_slice(&px.to_le_bytes());
                b.extend_from_slice(&py.to_le_bytes());
            }
            recs.push(b);
        }
    }
    let mut shp = Vec::new();
    let total: usize = 100 + recs.iter().map(|r| r.len() + 8).sum::<usize>();
    shp.extend_from_slice(&9994i32.to_be_bytes());
    shp.extend_from_slice(&[0u8; 20]);
    shp.extend_from_slice(&((total / 2) as i32).to_be_bytes());
    shp.extend_from_slice(&1000i32.to_le_bytes());
    shp.extend_from_slice(&5i32.to_le_bytes());
    shp.extend_from_slice(&[0u8; 64]);
    for (k, r) in recs.iter().enumerate() {
        shp.extend_from_slice(&((k + 1) as i32).to_be_bytes());
        shp.extend_from_slice(&((r.len() / 2) as i32).to_be_bytes());
        shp.extend_from_slice(r);
    }
    std::fs::write(base.with_extension("shp"), shp).unwrap();
    // dBase III with one character field NAME(10) and one numeric field POP(8)
    let n = recs.len();
    let mut dbf = vec![3u8, 124, 1, 1];
    dbf.extend_from_slice(&(n as u32).to_le_bytes());
    dbf.extend_from_slice(&(32u16 + 64 + 1).to_le_bytes());
    dbf.extend_from_slice(&(1u16 + 10 + 8).to_le_bytes());
    dbf.extend_from_slice(&[0u8; 20]);
    for (name, ty, len) in [("NAME", b'C', 10u8), ("POP", b'N', 8u8)] {
        let mut f = [0u8; 32];
        f[..name.len()].copy_from_slice(name.as_bytes());
        f[11] = ty;
        f[16] = len;
        dbf.extend_from_slice(&f);
    }
    dbf.push(0x0D);
    for k in 0..n {
        dbf.push(b' ');
        dbf.extend_from_slice(format!("{:<10}", format!("prov{}", k + 1)).as_bytes());
        dbf.extend_from_slice(format!("{:>8}", (k + 1) * 100).as_bytes());
    }
    dbf.push(0x1A);
    std::fs::write(base.with_extension("dbf"), dbf).unwrap();
}

#[test]
fn shp2dta_spmat_and_spmap() {
    let dir = temp_dir("spatial");
    let base = dir.join("grid");
    write_grid_shapefile(&base, 3, 2);
    let mut s = Session::new();
    let p = |name: &str| dir.join(name).display().to_string();
    s.run(&format!(
        "shp2dta using \"{}\", database(\"{}\") coordinates(\"{}\") genid(_ID) gencentroids(c) replace",
        base.with_extension("shp").display(),
        p("db"),
        p("coord")
    ));
    s.run(&format!("use \"{}\", clear", p("db.dta")));
    assert_eq!(s.n(), 6);
    assert_eq!(s.strs("NAME")[0], "prov1");
    assert_eq!(s.nums("POP")[5], 600.0);
    assert_eq!(s.nums("x_c")[0], 0.5);
    // queen contiguity on a 3x2 grid, row-normalised
    s.run(&format!("spmat contiguity W using \"{}\", id(_ID) normalize(row)", p("coord.dta")));
    s.run("spmat summarize W");
    assert_eq!(s.scalar("r(l_max)"), 5.0); // middle cells touch all five others
    assert_eq!(s.scalar("r(l_min)"), 3.0);
    s.run("spmat lag double Wpop W POP");
    // cell 1 (corner): neighbours 2, 4, 5 -> mean of 200, 400, 500
    assert_close(s.nums("Wpop")[0], (200.0 + 400.0 + 500.0) / 3.0, 1e-12);
    // choropleth and combined graph export
    s.run(&format!("spmap POP using \"{}\", id(_ID) clmethod(quantile) clnumber(3) fcolor(gs14 gs8 gs2) name(m1, replace)", p("coord.dta")));
    s.run(&format!("spmap Wpop using \"{}\", id(_ID) name(m2, replace)", p("coord.dta")));
    s.run("graph combine m1 m2, cols(2)");
    let png = dir.join("maps.png");
    s.run(&format!("graph export \"{}\", replace", png.display()));
    assert!(std::fs::metadata(&png).unwrap().len() > 1000);
    let pdf = dir.join("maps.pdf");
    s.run(&format!("graph export \"{}\", replace", pdf.display()));
    assert!(std::fs::read(&pdf).unwrap().starts_with(b"%PDF"));
}
