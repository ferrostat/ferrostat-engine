//! Time-series estimation and tests: newey, prais, arima, dfuller,
//! pperron, corrgram/wntestq, tssmooth and tsfilter. Reference values are
//! computed step by step with other commands or from closed forms.

use ferrostat_tests::*;

/// y_t = 1 + 0.5 x_t + u_t with AR(1) errors (rho = 0.6), 200 periods.
fn data() -> Session {
    let mut s = Session::new();
    s.script(
        "set obs 200\nset seed 2029\ngen t = _n\ntsset t\ngen x = rnormal()\ngen e = rnormal()\n\
         gen u = e in 1\nreplace u = 0.6*u[_n-1] + e in 2/l\ngen y = 1 + 0.5*x + u\n\
         gen w = 0 in 1\nreplace w = w[_n-1] + e in 2/l",
    );
    s
}

#[test]
fn operators_combine_and_expand() {
    let mut s = data();
    s.script("gen a = LD.y\ngen b = L.y - L2.y\ngen c = D2.y - (y - 2*L.y + L2.y)\ngen d = S2.y - (y - L2.y)");
    let (a, b) = (s.nums("a"), s.nums("b"));
    assert!(a[..2].iter().all(|v| v.is_nan()));
    assert_all_close(&a[2..], &b[2..], 1e-12);
    s.run("summarize c d");
    assert_close(s.scalar("r(max)"), 0.0, 1e-12);
    let out = s.run("regress y L(0/2).x");
    assert!(out.contains("L1.") && out.contains("L2."), "{out}");
    assert_close(s.scalar("e(N)"), 198.0, 0.0);
}

#[test]
fn newey_lag0_is_robust_and_hac_matches_formula() {
    let mut s = data();
    s.run("regress y x, vce(robust)");
    let se_r = s.scalar("_se[x]");
    let out = s.run("newey y x, lag(0)");
    assert!(out.contains("Regression with Newey–West standard errors") && out.contains("Maximum lag = 0"), "{out}");
    assert_close(s.scalar("_se[x]"), se_r, 1e-9);
    // intercept-only model: V = n/(n-1) * (1/n^2) * [sum e^2 + 2 sum_l (1 - l/(L+1)) sum e_t e_{t-l}]
    s.run("newey y, lag(2)");
    let se = s.scalar("_se[_cons]");
    s.script("summarize y, meanonly\ngen double r = y - r(mean)\ngen double p0 = r^2\ngen double p1 = r*L.r\ngen double p2 = r*L2.r");
    let tot = |s: &mut Session, v: &str| {
        s.run(&format!("summarize {v}, meanonly"));
        s.scalar("r(sum)")
    };
    let omega = tot(&mut s, "p0") + 2.0 * (2.0 / 3.0) * tot(&mut s, "p1") + 2.0 * (1.0 / 3.0) * tot(&mut s, "p2");
    let n = 200.0;
    assert_close(se, (n / (n - 1.0) * omega / (n * n)).sqrt(), 1e-9);
    assert_close(s.scalar("e(lag)"), 2.0, 0.0);
    s.fails("newey y x");
}

#[test]
fn prais_twostep_equals_manual_transform() {
    let mut s = data();
    s.script("quietly regress y x\npredict double r0, residuals\nquietly regress r0 L.r0, noconstant");
    let rho = s.scalar("_b[L.r0]");
    let out = s.run("prais y x, twostep");
    assert!(out.contains("Prais–Winsten AR(1) regression with twostep estimates") && out.contains("Durbin–Watson statistic (original)"), "{out}");
    assert_close(s.scalar("e(rho)"), rho, 1e-10);
    let (bx, bc, dw0) = (s.scalar("_b[x]"), s.scalar("_b[_cons]"), s.scalar("e(dw_0)"));
    s.script(&format!(
        "gen double ys = y - {rho}*L.y\ngen double xs = x - {rho}*L.x\ngen double cs = 1 - {rho}\n\
         replace ys = sqrt(1-{rho}^2)*y in 1\nreplace xs = sqrt(1-{rho}^2)*x in 1\nreplace cs = sqrt(1-{rho}^2) in 1\n\
         quietly regress ys xs cs, noconstant"
    ));
    assert_close(bx, s.scalar("_b[xs]"), 1e-9);
    assert_close(bc, s.scalar("_b[cs]"), 1e-9);
    // Durbin–Watson of the OLS residuals
    s.script("gen double dd = (r0 - L.r0)^2\ngen double ee = r0^2");
    s.run("summarize dd, meanonly");
    let num = s.scalar("r(sum)");
    s.run("summarize ee, meanonly");
    assert_close(dw0, num / s.scalar("r(sum)"), 1e-9);
    // iterated estimates converge to a fixed point near the true rho
    s.run("prais y x");
    let r_it = s.scalar("e(rho)");
    assert!((r_it - 0.6).abs() < 0.15, "{r_it}");
    s.run("prais y x, corc");
    assert_close(s.scalar("e(N)"), 199.0, 0.0);
}

#[test]
fn arima_regression_and_conditional_ar1() {
    let mut s = data();
    // no ARMA terms: ML equals OLS, sigma^2 = RSS/n
    s.run("regress y x");
    let (b, rss) = (s.scalar("_b[x]"), s.scalar("e(rss)"));
    s.run("arima y x");
    assert_close(s.scalar("_b[y:x]"), b, 1e-5);
    assert_close(s.scalar("e(sigma)"), (rss / 200.0).sqrt(), 1e-5);
    // conditional ML of an AR(1) is least squares on the lag
    s.run("regress u L.u, noconstant");
    let (phi, rss) = (s.scalar("_b[L.u]"), s.scalar("e(rss)"));
    let out = s.run("arima u, ar(1) noconstant condition");
    assert!(out.contains("ARIMA regression") && out.contains("/sigma"), "{out}");
    assert_close(s.scalar("_b[ARMA:L.ar]"), phi, 1e-5);
    assert_close(s.scalar("e(sigma)"), (rss / 199.0).sqrt(), 1e-5);
}

#[test]
fn arima_exact_ar1_likelihood_and_predictions() {
    let mut s = data();
    s.run("arima u, ar(1) noconstant");
    let (phi, sig) = (s.scalar("_b[ARMA:L.ar]"), s.scalar("e(sigma)"));
    assert!((phi - 0.6).abs() < 0.15);
    // exact AR(1) log likelihood at the estimates
    s.script(&format!("gen double q = (u - {phi}*L.u)^2\nreplace q = (1-{phi}^2)*u^2 in 1"));
    s.run("summarize q, meanonly");
    let n = 200.0;
    let ll = -0.5 * n * (2.0 * std::f64::consts::PI * sig * sig).ln() + 0.5 * (1.0 - phi * phi).ln() - s.scalar("r(sum)") / (2.0 * sig * sig);
    assert_close(s.scalar("e(ll)"), ll, 1e-8);
    // the maximum: score of phi is zero (ll at nearby phi is lower)
    s.run(&format!("quietly arima u, ar(1) noconstant from({} {sig}) iterate(0)", phi + 0.01));
    assert!(s.scalar("e(ll)") < ll);
    // one-step predictions phi*u_{t-1}; MSE sigma^2/(1-phi^2) at t = 1
    s.run("quietly arima u, ar(1) noconstant");
    s.script("predict double p\npredict double m, mse\npredict double r, residuals");
    let (u, p, m, r) = (s.nums("u"), s.nums("p"), s.nums("m"), s.nums("r"));
    assert_close(p[0], 0.0, 1e-12);
    assert_close(m[0], sig * sig / (1.0 - phi * phi), 1e-8);
    for t in 1..200 {
        assert_close(p[t], phi * u[t - 1], 1e-8);
        assert_close(m[t], sig * sig, 1e-8);
        assert_close(r[t], u[t] - p[t], 1e-8);
    }
    let out = s.run("estat aroots");
    assert!(out.contains("AR parameters satisfy stability condition."), "{out}");
    assert_close(s.scalar("el(r(Modulus_ar),1,1)"), phi.abs(), 1e-6);
    s.run("estat ic");
    assert_close(s.scalar("el(r(S),1,4)"), 2.0, 0.0);
}

#[test]
fn arima_differencing_and_levels() {
    let mut s = data();
    // ARIMA(0,1,0) with drift: _cons is the mean of D.w; y predictions add w_{t-1}
    let out = s.run("arima w, arima(0,1,0)");
    assert!(out.contains("D.w") && out.contains("Sample: 2 thru 200"), "{out}");
    s.script("gen double dw = D.w\nsummarize dw");
    assert_close(s.scalar("_b[w:_cons]"), s.scalar("r(mean)"), 1e-5);
    s.run("quietly arima w, arima(0,1,0)");
    s.script("predict double lv, y\npredict double dx");
    let (w, lv, dx) = (s.nums("w"), s.nums("lv"), s.nums("dx"));
    for t in 1..200 {
        assert_close(lv[t], w[t - 1] + dx[t], 1e-9);
    }
    // ARMA(1,1) recovers the AR(1) error process reasonably
    s.run("quietly arima u, arima(1,0,1)");
    assert!((s.scalar("_b[ARMA:L.ar]") - 0.6).abs() < 0.25);
    assert_close(s.scalar("e(k)"), 4.0, 0.0);
}

#[test]
fn dfuller_and_pperron_statistics() {
    let mut s = data();
    s.run("regress D.w L.w");
    let t = s.scalar("_b[L.w]/_se[L.w]");
    let out = s.run("dfuller w");
    assert!(out.contains("Dickey–Fuller test for unit root") && out.contains("MacKinnon approximate p-value"), "{out}");
    assert_close(s.scalar("r(Zt)"), t, 1e-9);
    assert_close(s.scalar("r(N)"), 199.0, 0.0);
    let p = s.scalar("r(p)");
    assert!(p > 0.0 && p < 1.0);
    // Fuller table interpolated linearly in N (N = 199, constant)
    assert_close(s.scalar("r(cv_5)"), -2.89 + (99.0 / 150.0) * 0.01, 1e-9);
    s.run("dfuller w, lags(2) trend regress");
    s.run("regress D.w L.w LD.w L2D.w t");
    let t2 = s.scalar("_b[L.w]/_se[L.w]");
    s.run("dfuller w, lags(2) trend");
    assert_close(s.scalar("r(Zt)"), t2, 1e-9);
    // a stationary series rejects strongly
    s.run("dfuller u");
    assert!(s.scalar("r(p)") < 0.01);
    // pperron with no Newey–West lags is the Dickey–Fuller regression
    s.run("regress w L.w");
    let (rho, se) = (s.scalar("_b[L.w]"), s.scalar("_se[L.w]"));
    s.run("pperron w, lags(0)");
    assert_close(s.scalar("r(Zt)"), (rho - 1.0) / se, 1e-9);
    assert_close(s.scalar("r(Zrho)"), 199.0 * (rho - 1.0), 1e-9);
    s.run("pperron w");
    assert_close(s.scalar("r(lags)"), (4.0f64 * (199.0f64 / 100.0).powf(2.0 / 9.0)).floor(), 0.0);
}

#[test]
fn corrgram_and_portmanteau() {
    let mut s = data();
    let out = s.run("corrgram u, lags(4)");
    assert!(out.contains("[Autocorrelation]") && out.lines().count() >= 7, "{out}");
    let ac1 = s.scalar("r(ac1)");
    let q2 = s.scalar("r(q2)");
    let ac2 = s.scalar("r(ac2)");
    let pac1 = s.scalar("r(pac1)");
    s.script("summarize u, meanonly\ngen double d = u - r(mean)\ngen double c1 = d*L.d\ngen double c0 = d^2");
    s.run("summarize c1, meanonly");
    let num = s.scalar("r(sum)");
    s.run("summarize c0, meanonly");
    assert_close(ac1, num / s.scalar("r(sum)"), 1e-10);
    let n = 200.0;
    assert_close(q2, n * (n + 2.0) * (ac1 * ac1 / (n - 1.0) + ac2 * ac2 / (n - 2.0)), 1e-9);
    s.run("regress u L.u");
    assert_close(pac1, s.scalar("_b[L.u]"), 1e-9);
    s.run("wntestq u, lags(4)");
    let q4 = s.scalar("r(stat)");
    s.run("corrgram u, lags(4) noplot");
    assert_close(q4, s.scalar("r(q4)"), 1e-12);
}

#[test]
fn tssmooth_recursions() {
    let mut s = data();
    s.run("tssmooth ma m3 = x, window(1 1 1)");
    s.run("gen double chk = (L.x + x + F.x)/3");
    let (m3, chk) = (s.nums("m3"), s.nums("chk"));
    assert_all_close(&m3[1..199], &chk[1..199], 1e-12);
    s.run("tssmooth ma mw = x, weights(1 <2> 1)");
    let (mw, x) = (s.nums("mw"), s.nums("x"));
    assert_close(mw[10], (x[9] + 2.0 * x[10] + x[11]) / 4.0, 1e-12);
    // single exponential with s0 = mean of the first 100 observations
    s.run("tssmooth exponential ex = x, parms(.3)");
    assert_close(s.scalar("r(alpha)"), 0.3, 0.0);
    s.run("summarize x in 1/100, meanonly");
    let mut sm = s.scalar("r(mean)");
    for v in &x {
        sm = 0.3 * v + 0.7 * sm;
    }
    assert_close(*s.nums("ex").last().unwrap(), sm, 1e-10);
    // the optimal coefficient has no smaller SSE than fixed ones
    s.run("tssmooth exponential ex2 = w");
    let best = s.scalar("r(rss)");
    s.run("tssmooth exponential ex3 = w, parms(.5)");
    assert!(best <= s.scalar("r(rss)") + 1e-9);
    s.run("tssmooth hwinters hw = w, parms(.5 .2)");
    assert!(s.scalar("r(rmse)") > 0.0);
    s.run("tssmooth nl nl1 = x, smoother(3)");
    let nl = s.nums("nl1");
    let mut v = [x[4], x[5], x[6]];
    v.sort_by(f64::total_cmp);
    assert_close(nl[5], v[1], 1e-12);
    s.fails("tssmooth ma m3 = x, window(1 1 1)");
}

#[test]
fn tsfilter_kills_linear_trends() {
    let mut s = data();
    s.run("gen double lin = 3 + 0.25*t");
    for f in ["hp", "bk", "cf", "bw"] {
        let opt = if f == "cf" { ", drift" } else { "" };
        s.run(&format!("tsfilter {f} c_{f} = lin{opt}"));
        let c: Vec<f64> = s.nums(&format!("c_{f}")).into_iter().filter(|v| v.is_finite()).collect();
        assert!(c.iter().all(|v| v.abs() < 1e-6), "{f}: {:?}", &c[..3]);
    }
    // HP: trend + cycle = series, and the cycle sums to zero
    s.run("tsfilter hp cyc = w, smooth(1600) trend(tr)");
    s.run("gen double chk = cyc + tr - w");
    s.run("summarize chk");
    assert_close(s.scalar("r(max)"), 0.0, 1e-9);
    s.run("summarize cyc, meanonly");
    assert_close(s.scalar("r(sum)"), 0.0, 1e-7);
    // Baxter–King loses smaorder() observations at each end
    s.run("tsfilter bk bkc = w, smaorder(8)");
    s.run("count if bkc < .");
    assert_close(s.scalar("r(N)"), 184.0, 0.0);
}
