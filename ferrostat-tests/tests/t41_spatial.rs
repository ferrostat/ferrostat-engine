//! Spatial autoregressive models: spset/spshape2dta, spmatrix (contiguity,
//! inverse distance, save/use, export/import), spgenerate, spregress (ML and
//! GS2SLS), estat impact, predict, estat moran and spxtregress.
//!
//! Checks are against independent computations: known contiguity counts of
//! a lattice, a concentrated likelihood maximized by hand, two-stage least
//! squares with the spatial instruments, and Moran's statistic computed from
//! the residuals.

use std::path::Path;

use ferrostat_tests::*;

/// Shapefile of `cols` x `rows` unit squares with NAME and POP attributes.
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

/// Session with a `side` x `side` lattice loaded and spset.
fn lattice(test: &str, side: usize) -> (Session, String) {
    let dir = temp_dir(test);
    let base = dir.join("grid");
    write_grid_shapefile(&base, side, side);
    let mut s = Session::new();
    s.run(&format!("cd \"{}\"", dir.display()));
    let out = s.run("spshape2dta grid, replace");
    assert!(out.contains("grid_shp.dta") && out.contains("grid.dta"), "{out}");
    s.run("use grid, clear");
    (s, dir.display().to_string())
}

/// Rook neighbours on a side x side lattice (IDs row-major from 1).
fn rook(side: usize) -> Vec<Vec<f64>> {
    let n = side * side;
    let mut w = vec![vec![0.0; n]; n];
    for r in 0..side {
        for c in 0..side {
            let i = r * side + c;
            if c + 1 < side {
                w[i][i + 1] = 1.0;
                w[i + 1][i] = 1.0;
            }
            if r + 1 < side {
                w[i][i + side] = 1.0;
                w[i + side][i] = 1.0;
            }
        }
    }
    w
}

fn row_normalize(w: &mut [Vec<f64>]) {
    for r in w.iter_mut() {
        let s: f64 = r.iter().sum();
        r.iter_mut().for_each(|v| *v /= s);
    }
}

fn det(mut a: Vec<Vec<f64>>) -> f64 {
    let n = a.len();
    let mut d = 1.0;
    for k in 0..n {
        let p = (k..n).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs())).unwrap();
        if p != k {
            a.swap(p, k);
            d = -d;
        }
        d *= a[k][k];
        for i in k + 1..n {
            let f = a[i][k] / a[k][k];
            for j in k..n {
                a[i][j] -= f * a[k][j];
            }
        }
    }
    d
}

fn matvec(w: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    w.iter().map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum()).collect()
}

/// OLS of y on [1, x]: (b0, b1, ssr).
fn ols(y: &[f64], x: &[f64]) -> (f64, f64, f64) {
    let n = y.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    let b1 = sxy / sxx;
    let b0 = my - b1 * mx;
    let ssr = x.iter().zip(y).map(|(a, b)| (b - b0 - b1 * a).powi(2)).sum();
    (b0, b1, ssr)
}

#[test]
fn spset_and_contiguity_counts() {
    let (mut s, _) = lattice("t41_contig", 3);
    let out = s.run("spset");
    assert!(out.contains("Spatial-unit ID: _ID") && out.contains("grid_shp.dta") && out.contains("_CX, _CY (planar)"), "{out}");
    // centroids of the unit squares
    assert_all_close(&s.nums("_CX")[..3], &[0.5, 1.5, 2.5], 1e-12);
    assert_close(s.nums("_CY")[4], 1.5, 1e-12);
    // queen: corners 3, edges 5, center 8 neighbours
    s.run("spmatrix create contiguity W, normalize(none)");
    s.run("spmatrix summarize W, generate(nb)");
    assert_eq!(s.scalar("r(n_min)"), 3.0);
    assert_eq!(s.scalar("r(n_max)"), 8.0);
    assert_close(s.scalar("r(n_mean)"), 40.0 / 9.0, 1e-12);
    assert_eq!(s.nums("nb"), vec![3.0, 5.0, 3.0, 5.0, 8.0, 5.0, 3.0, 5.0, 3.0]);
    // rook: corners 2, edges 3, center 4
    s.run("spmatrix create contiguity R, rook normalize(none)");
    s.run("spmatrix summarize R");
    assert_eq!(s.scalar("r(n_min)"), 2.0);
    assert_eq!(s.scalar("r(n_max)"), 4.0);
    assert_close(s.scalar("r(n_mean)"), 24.0 / 9.0, 1e-12);
    // second-order rook neighbours of the corner: (0,2), (1,1), (2,0)
    s.run("spmatrix create contiguity R2, rook second normalize(none)");
    s.run("spmatrix matafromsp M id = R2");
    let row1: Vec<f64> = (1..=9).map(|j| s.scalar(&format!("el(M,1,{j})"))).collect();
    assert_eq!(row1, vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
    // spectral normalization: largest eigenvalue of the queen 3x3 lattice
    s.run("spmatrix create contiguity S");
    s.run("spmatrix summarize S");
    let lmax = s.scalar("r(max)");
    // queen 3x3 adjacency has spectral radius 1 + sqrt(17) ... check via scaling of a 0/1 matrix
    assert!(lmax > 0.0 && lmax < 1.0, "{lmax}");
    // spatial lag of x with W (row sums)
    s.run("gen one = 1");
    s.run("spgenerate wone = W*one");
    assert_eq!(s.nums("wone"), vec![3.0, 5.0, 3.0, 5.0, 8.0, 5.0, 3.0, 5.0, 3.0]);
    // matrix dir does not list the weighting matrices
    let dir = s.run("matrix dir");
    assert!(!dir.contains("__spmatrix"), "{dir}");
    let out = s.run("spmatrix dir");
    assert!(out.contains("contiguity") && out.contains(" R2 ") || out.contains("R2"), "{out}");
}

#[test]
fn spmatrix_files_and_utilities() {
    let (mut s, dir) = lattice("t41_files", 3);
    s.run("spmatrix create idistance D, normalize(none)");
    s.run("spmatrix matafromsp A ida = D");
    // distance between centroids of units 1 and 2 is 1, between 1 and 5 is sqrt(2)
    assert_close(s.scalar("el(A,1,2)"), 1.0, 1e-12);
    assert_close(s.scalar("el(A,1,5)"), 1.0 / 2f64.sqrt(), 1e-12);
    s.run("spmatrix create idistance Dt, normalize(none) vtruncate(0.8)");
    s.run("spmatrix matafromsp B idb = Dt");
    assert_eq!(s.scalar("el(B,1,5)"), 0.0);
    assert_eq!(s.scalar("el(B,1,2)"), 1.0);
    let out = s.run("spdistance 1 5");
    assert!(out.contains("distance"), "{out}");
    assert_close(s.scalar("r(distance)"), 2f64.sqrt(), 1e-12);
    // save/use round trip is exact; export/import keeps values
    s.run("spmatrix create contiguity W");
    s.run("spmatrix note W : queen contiguity of the test lattice");
    s.run(&format!("spmatrix save W using \"{dir}/w\", replace"));
    s.run(&format!("spmatrix export W using \"{dir}/w.txt\", replace"));
    s.run("spmatrix copy W W0");
    s.run("spmatrix drop W");
    s.fails("spmatrix summarize W");
    s.run(&format!("spmatrix use W using \"{dir}/w\""));
    s.run(&format!("spmatrix import Wi using \"{dir}/w.txt\""));
    s.run("spmatrix note W");
    assert_eq!(s.run("display r(note)"), "queen contiguity of the test lattice");
    s.run("spmatrix matafromsp M1 i1 = W");
    s.run("spmatrix matafromsp M0 i0 = W0");
    s.run("spmatrix matafromsp M2 i2 = Wi");
    for (i, j) in [(1, 2), (5, 9), (3, 6)] {
        let a = s.scalar(&format!("el(M0,{i},{j})"));
        assert_eq!(s.scalar(&format!("el(M1,{i},{j})")), a);
        assert_close(s.scalar(&format!("el(M2,{i},{j})")), a, 1e-15);
    }
    // spfrommata and normalize
    s.run("spmatrix spfrommata Wr = M0 i0, normalize(row)");
    s.run("spmatrix summarize Wr");
    assert_close(s.scalar("r(max)"), 1.0 / 3.0, 1e-12);
    s.run("spmatrix normalize W0, normalize(row)");
    s.run("spmatrix matafromsp M3 i3 = W0");
    assert_close(s.scalar("el(M3,5,1)"), 0.125, 1e-12);
    // fromdata: a full matrix stored in variables
    s.run("gen c1 = _ID == 2");
    s.run("gen c2 = _ID == 1");
    s.run("forvalues k = 3/9 {\n gen c`k' = 0\n}");
    s.run("spmatrix fromdata F = c1-c9, normalize(none)");
    s.run("spmatrix matafromsp MF idf = F");
    assert_eq!(s.scalar("el(MF,1,2)"), 1.0);
    assert_eq!(s.scalar("el(MF,2,1)"), 1.0);
    assert_eq!(s.scalar("el(MF,3,4)"), 0.0);
}

/// Data on a 5x5 lattice with spatially correlated outcome.
fn sar_data(test: &str) -> (Session, Vec<f64>, Vec<f64>) {
    let (mut s, _) = lattice(test, 5);
    s.run("gen x = sin(_ID*1.3) + _ID/10");
    s.run("gen y0 = 1 + 2*x + cos(1.7*_ID)");
    s.run("spmatrix create contiguity W, rook normalize(row)");
    s.run("spgenerate wy0 = W*y0");
    s.run("gen y = y0 + 0.6*wy0");
    let (x, y) = (s.nums("x"), s.nums("y"));
    (s, x, y)
}

#[test]
fn sar_ml_matches_hand_maximized_likelihood() {
    let (mut s, x, y) = sar_data("t41_ml");
    let mut w = rook(5);
    row_normalize(&mut w);
    let n = y.len() as f64;
    let wy = matvec(&w, &y);
    let conc = |lam: f64| {
        let yl: Vec<f64> = y.iter().zip(&wy).map(|(a, b)| a - lam * b).collect();
        let (_, _, ssr) = ols(&yl, &x);
        let mut a: Vec<Vec<f64>> = w.iter().map(|r| r.iter().map(|v| -lam * v).collect()).collect();
        for i in 0..a.len() {
            a[i][i] += 1.0;
        }
        -n / 2.0 * ((2.0 * std::f64::consts::PI * ssr / n).ln() + 1.0) + det(a).ln()
    };
    // golden section on (-0.99, 0.99)
    let (mut lo, mut hi) = (-0.99f64, 0.99f64);
    let g = (5f64.sqrt() - 1.0) / 2.0;
    for _ in 0..200 {
        let (c, d) = (hi - g * (hi - lo), lo + g * (hi - lo));
        if conc(c) > conc(d) { hi = d } else { lo = c }
    }
    let lam = (lo + hi) / 2.0;
    let out = s.run("spregress y x, ml dvarlag(W)");
    assert!(out.contains("Spatial autoregressive model") && out.contains("Maximum likelihood estimates") && out.contains("var(e.y)"), "{out}");
    assert!(out.contains("Wald test of spatial terms"), "{out}");
    assert_close(s.scalar("_b[W:y]"), lam, 1e-6);
    assert_close(s.scalar("e(ll)"), conc(lam), 1e-8);
    let yl: Vec<f64> = y.iter().zip(&wy).map(|(a, b)| a - lam * b).collect();
    let (b0, b1, ssr) = ols(&yl, &x);
    assert_close(s.scalar("_b[y:x]"), b1, 1e-5);
    assert_close(s.scalar("_b[y:_cons]"), b0, 1e-5);
    assert_close(s.scalar("e(sigma2)"), ssr / n, 1e-6);
    assert_eq!(s.run("display e(cmd)"), "spregress");
    assert_eq!(s.run("display e(estimator)"), "ml");
    assert_eq!(s.scalar("e(N)"), 25.0);
    // standard errors are positive and the spatial Wald test has 1 df
    assert!(s.scalar("_se[W:y]") > 0.0);
    assert_eq!(s.scalar("e(df_c)"), 1.0);
    // impacts: with a row-normalized W, total = beta/(1-lambda)
    let out = s.run("estat impact");
    assert!(out.contains("direct") && out.contains("indirect") && out.contains("total") && out.contains("Delta-Method"), "{out}");
    let bl = s.scalar("_b[W:y]");
    let bx = s.scalar("_b[y:x]");
    s.run("matrix T_ = r(b_total)");
    s.run("matrix D_ = r(b_direct)");
    s.run("matrix I_ = r(b_indirect)");
    assert_close(s.scalar("el(T_,1,1)"), bx / (1.0 - bl), 1e-8);
    let d = s.scalar("el(D_,1,1)");
    let ind = s.scalar("el(I_,1,1)");
    assert_close(d + ind, bx / (1.0 - bl), 1e-10);
    assert!(d > bx && ind > 0.0);
    // predictions: (I - lambda W) rform = xb
    s.run("predict rf");
    s.run("predict double xbv, xb");
    s.run("predict double nv, naive");
    s.run("predict double dv, direct");
    s.run("predict double iv, indirect");
    s.run("predict double rfd, rform");
    s.run("spgenerate double wrf = W*rfd");
    let (rf, wrf, xb) = (s.nums("rfd"), s.nums("wrf"), s.nums("xbv"));
    for i in 0..25 {
        assert_close(rf[i] - bl * wrf[i], xb[i], 1e-8);
    }
    let (dv, iv) = (s.nums("dv"), s.nums("iv"));
    for i in 0..25 {
        assert_close(dv[i] + iv[i], rf[i], 1e-8);
    }
    let nv = s.nums("nv");
    assert_close(nv[3], xb[3] + bl * wy[3], 1e-8);
}

#[test]
fn sarar_ml_and_error_model() {
    let (mut s, x, y) = sar_data("t41_sarar");
    let mut w = rook(5);
    row_normalize(&mut w);
    let n = y.len() as f64;
    // SEM: hand concentrated likelihood in rho
    let my = matvec(&w, &y);
    let mx = matvec(&w, &x);
    let conc = |rho: f64| {
        let yt: Vec<f64> = y.iter().zip(&my).map(|(a, b)| a - rho * b).collect();
        let xt: Vec<f64> = x.iter().zip(&mx).map(|(a, b)| a - rho * b).collect();
        // regress yt on (1 - rho) and xt
        let c: Vec<f64> = vec![1.0 - rho; x.len()];
        let (s11, s12, s22) = (c.iter().map(|v| v * v).sum::<f64>(), c.iter().zip(&xt).map(|(a, b)| a * b).sum::<f64>(), xt.iter().map(|v| v * v).sum::<f64>());
        let (r1, r2) = (c.iter().zip(&yt).map(|(a, b)| a * b).sum::<f64>(), xt.iter().zip(&yt).map(|(a, b)| a * b).sum::<f64>());
        let dd = s11 * s22 - s12 * s12;
        let (b0, b1) = ((s22 * r1 - s12 * r2) / dd, (s11 * r2 - s12 * r1) / dd);
        let ssr: f64 = (0..x.len()).map(|i| (yt[i] - b0 * c[i] - b1 * xt[i]).powi(2)).sum();
        let mut a: Vec<Vec<f64>> = w.iter().map(|r| r.iter().map(|v| -rho * v).collect()).collect();
        for i in 0..a.len() {
            a[i][i] += 1.0;
        }
        -n / 2.0 * ((2.0 * std::f64::consts::PI * ssr / n).ln() + 1.0) + det(a).ln()
    };
    let (mut lo, mut hi) = (-0.99f64, 0.99f64);
    let g = (5f64.sqrt() - 1.0) / 2.0;
    for _ in 0..200 {
        let (c, d) = (hi - g * (hi - lo), lo + g * (hi - lo));
        if conc(c) > conc(d) { hi = d } else { lo = c }
    }
    let rho = (lo + hi) / 2.0;
    s.run("spregress y x, ml errorlag(W)");
    assert_close(s.scalar("_b[W:e.y]"), rho, 1e-6);
    assert_close(s.scalar("e(ll)"), conc(rho), 1e-8);
    // SARAR with an exogenous lag: likelihood at least as large as SAR alone
    s.run("spregress y x, ml dvarlag(W)");
    let ll_sar = s.scalar("e(ll)");
    let out = s.run("spregress y x, ml dvarlag(W) errorlag(W) ivarlag(W: x)");
    assert!(out.contains("e.y") && out.contains("Wald test of spatial terms:          chi2(3)"), "{out}");
    assert!(s.scalar("e(ll)") >= ll_sar - 1e-8);
    assert_eq!(s.run("display e(model)"), "sarar");
    s.run("estat impact x");
    assert!({ s.run("matrix T_ = r(b_total)"); s.scalar("el(T_,1,1)") }.is_finite());
}

#[test]
fn gs2sls_equals_spatial_two_stage_least_squares() {
    let (mut s, _, _) = sar_data("t41_gs2sls");
    s.run("spgenerate double wy = W*y");
    s.run("spgenerate double wx = W*x");
    s.run("spgenerate double w2x = W*wx");
    s.run("ivregress 2sls y x (wy = wx w2x)");
    let (b_wy, se_wy, b_x, se_x) = (s.scalar("_b[wy]"), s.scalar("_se[wy]"), s.scalar("_b[x]"), s.scalar("_se[x]"));
    let out = s.run("spregress y x, gs2sls dvarlag(W)");
    assert!(out.contains("GS2SLS estimates") && out.contains("Pseudo R2"), "{out}");
    assert_close(s.scalar("_b[W:y]"), b_wy, 1e-9);
    assert_close(s.scalar("_b[y:x]"), b_x, 1e-9);
    assert_close(s.scalar("_se[W:y]"), se_wy, 1e-8);
    assert_close(s.scalar("_se[y:x]"), se_x, 1e-8);
    // heteroskedastic: robust 2SLS
    s.run("ivregress 2sls y x (wy = wx w2x), vce(robust)");
    let se_r = s.scalar("_se[wy]");
    s.run("spregress y x, gs2sls dvarlag(W) heteroskedastic");
    assert_close(s.scalar("_se[W:y]"), se_r, 1e-8);
    // with a spatial error: rho is estimated and reported under W
    let out = s.run("spregress y x, gs2sls dvarlag(W) errorlag(W)");
    assert!(out.contains("e.y"), "{out}");
    let r = s.scalar("_b[W:e.y]");
    assert!(r.abs() < 1.0 && s.scalar("_se[W:e.y]") > 0.0, "{r}");
    s.run("spregress y x, gs2sls errorlag(W) heteroskedastic");
    assert!(s.scalar("_se[W:e.y]") > 0.0);
    // spivregress with an external instrument
    s.run("gen z = cos(_ID)");
    s.run("gen xe = x + 0.3*z");
    let out = s.run("spivregress y (xe = z), dvarlag(W)");
    assert!(out.contains("xe"), "{out}");
    assert_eq!(s.run("display e(cmd)"), "spivregress");
}

#[test]
fn moran_test_after_regress() {
    let (mut s, x, y) = sar_data("t41_moran");
    s.run("spmatrix create contiguity B, rook normalize(none)");
    s.run("regress y x");
    let out = s.run("estat moran, errorlag(B)");
    assert!(out.contains("Moran test for spatial dependence") && out.contains("chi2(1)"), "{out}");
    let (b0, b1, ssr) = ols(&y, &x);
    let e: Vec<f64> = x.iter().zip(&y).map(|(a, b)| b - b0 - b1 * a).collect();
    let w = rook(5);
    let n = e.len() as f64;
    let q = e.iter().zip(matvec(&w, &e)).map(|(a, b)| a * b).sum::<f64>() / (ssr / n);
    let links: f64 = w.iter().flatten().sum();
    let chi2 = q * q / (2.0 * links);
    assert_close(s.scalar("r(chi2)"), chi2, 1e-9);
    assert_eq!(s.scalar("r(df)"), 1.0);
    s.run("estat moran, errorlag(B) errorlag(W)");
    assert_eq!(s.scalar("r(df)"), 2.0);
}

#[test]
fn spxtregress_fixed_and_random_effects() {
    let (mut s, _) = lattice("t41_xt", 4);
    s.run("keep _ID");
    s.run("expand 3");
    s.run("bysort _ID: gen t = _n");
    s.run("xtset _ID t");
    s.run("spset _ID");
    s.run("spset, modify shpfile(grid_shp)");
    let out = s.run("spbalance");
    assert!(out.contains("strongly balanced"), "{out}");
    s.run("spmatrix create contiguity W, rook normalize(row)");
    s.run("gen x = sin(_ID*0.7 + t) + t/5");
    s.run("gen a = cos(_ID)");
    s.run("gen y0 = a + 1.5*x + 0.5*sin(3*_ID*t)");
    s.run("spgenerate wy0 = W*y0");
    s.run("gen y = y0 + 0.4*wy0");
    let out = s.run("spxtregress y x, fe dvarlag(W)");
    assert!(out.contains("Fixed-effects spatial regression") && out.contains("/sigma_e"), "{out}");
    let lam = s.scalar("_b[W:y]");
    assert!(lam > 0.0 && lam < 1.0, "{lam}");
    // log likelihood at the estimates equals the Lee–Yu formula
    let (n, t) = (16usize, 3usize);
    let mut w = rook(4);
    row_normalize(&mut w);
    let ids = s.nums("_ID");
    let ts = s.nums("t");
    let yv = s.nums("y");
    let xv = s.nums("x");
    let mut yb = vec![vec![0.0; n]; t];
    let mut xbm = vec![vec![0.0; n]; t];
    for k in 0..ids.len() {
        yb[ts[k] as usize - 1][ids[k] as usize - 1] = yv[k];
        xbm[ts[k] as usize - 1][ids[k] as usize - 1] = xv[k];
    }
    let mean = |m: &Vec<Vec<f64>>, i: usize| (0..t).map(|p| m[p][i]).sum::<f64>() / t as f64;
    let bx = s.scalar("_b[y:x]");
    let se = s.scalar("_b[/:sigma_e]");
    let mut ssr = 0.0;
    for p in 0..t {
        let yd: Vec<f64> = (0..n).map(|i| yb[p][i] - mean(&yb, i)).collect();
        let xd: Vec<f64> = (0..n).map(|i| xbm[p][i] - mean(&xbm, i)).collect();
        let wyd = matvec(&w, &yd);
        for i in 0..n {
            ssr += (yd[i] - lam * wyd[i] - bx * xd[i]).powi(2);
        }
    }
    let mut a: Vec<Vec<f64>> = w.iter().map(|r| r.iter().map(|v| -lam * v).collect()).collect();
    for i in 0..n {
        a[i][i] += 1.0;
    }
    let neff = (n * (t - 1)) as f64;
    let ll = -neff / 2.0 * (2.0 * std::f64::consts::PI * se * se).ln() + (t - 1) as f64 * det(a).ln() - ssr / (2.0 * se * se);
    assert_close(s.scalar("e(ll)"), ll, 1e-7);
    assert_close(se * se, ssr / neff, 1e-6);
    let out = s.run("spxtregress y x, re dvarlag(W)");
    assert!(out.contains("Random-effects spatial regression") && out.contains("/sigma_u"), "{out}");
    assert!(s.scalar("_b[/:sigma_u]") >= 0.0 && s.scalar("_b[/:sigma_e]") > 0.0);
    assert_eq!(s.scalar("e(N_g)"), 16.0);
    s.run("estat impact");
    assert!({ s.run("matrix T_ = r(b_total)"); s.scalar("el(T_,1,1)") }.is_finite());
    // unbalanced panels are reported
    s.run("drop if _ID == 3 & t == 2");
    let out = s.run("spbalance");
    assert!(out.contains("not strongly balanced"), "{out}");
    s.run("spbalance, balance");
    assert_eq!(s.n(), 45);
}

#[test]
fn grmap_draws_a_choropleth() {
    let (mut s, _) = lattice("t41_grmap", 3);
    s.run("gen v = _ID");
    s.run("grmap v, clnumber(3)");
    let out = s.run("graph describe");
    assert!(!out.is_empty());
    s.run("spset, clear");
    s.fails("grmap v");
}
