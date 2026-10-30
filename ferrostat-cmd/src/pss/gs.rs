//! Group-sequential designs: `gsbounds` (boundaries only) and `gsdesign
//! <method>` (boundaries plus the maximum and per-look sample sizes).
//!
//! The standardized statistics Z_k = S(t_k)/√t_k are built from a Brownian
//! motion with drift θ on the information scale t ∈ (0, 1]; boundary
//! crossing probabilities come from recursive numerical integration of the
//! sub-density of Z_k on the continuation region (composite Simpson rule on
//! a fine grid), stage by stage.

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use super::ncdist::zcrit;
use super::{g, grid, has, positional, Row};
use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::matrix::Matrix;
use crate::store::{Results, Store};

// ---------------------------------------------------------------------------
// recursive integration
// ---------------------------------------------------------------------------

/// Sub-density of Z at the previous look on its grid: points and weights
/// (quadrature weight × density), at information `t`.
#[derive(Clone)]
pub struct State {
    z: Vec<f64>,
    w: Vec<f64>,
    t: f64,
}

impl State {
    pub fn start() -> State {
        State { z: vec![0.0], w: vec![1.0], t: 0.0 }
    }

    fn mass_cdf(&self, tk: f64, theta: f64, b: f64) -> f64 {
        // Σ w_i P(Z_k ≤ b | Z_{k-1} = z_i)
        if b == f64::INFINITY {
            return self.w.iter().sum();
        }
        if b == f64::NEG_INFINITY {
            return 0.0;
        }
        let d = tk - self.t;
        let sd = d.sqrt();
        let (sk, sp) = (tk.sqrt(), self.t.sqrt());
        self.z.iter().zip(&self.w).map(|(z, w)| w * dist::normal((b * sk - z * sp - theta * d) / sd)).sum()
    }

    /// P(continue to here and a < Z_k < b).
    pub fn prob(&self, tk: f64, theta: f64, a: f64, b: f64) -> f64 {
        (self.mass_cdf(tk, theta, b) - self.mass_cdf(tk, theta, a)).max(0.0)
    }

    /// Sub-density at the next look, restricted to the continuation intervals.
    pub fn advance(&self, tk: f64, theta: f64, intervals: &[(f64, f64)]) -> State {
        let d = tk - self.t;
        let sd = d.sqrt();
        let (sk, sp) = (tk.sqrt(), self.t.sqrt());
        // kernel width in z units sets the grid spacing
        let h = (sd / sk / 8.0).min(0.04);
        let mut z = Vec::new();
        let mut w = Vec::new();
        let lo_trunc = (theta * sk).min(0.0) - 9.0;
        let hi_trunc = (theta * sk).max(0.0) + 9.0;
        for &(a, b) in intervals {
            let (a, b) = (a.max(lo_trunc), b.min(hi_trunc));
            if !(b > a) {
                continue;
            }
            let mut n = ((b - a) / h).ceil() as usize;
            n = n.max(2);
            if n % 2 == 1 {
                n += 1;
            }
            let step = (b - a) / n as f64;
            for i in 0..=n {
                let x = a + i as f64 * step;
                let sw = if i == 0 || i == n { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
                let dens: f64 = self.z.iter().zip(&self.w).map(|(zi, wi)| wi * dist::normalden((x * sk - zi * sp - theta * d) / sd)).sum::<f64>() * sk / sd;
                z.push(x);
                w.push(sw * step / 3.0 * dens);
            }
        }
        State { z, w, t: tk }
    }
}

// ---------------------------------------------------------------------------
// boundary families
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum Family {
    None,
    /// classical Wang–Tsiatis family c_k = C·t_k^(Δ−1/2) (OBF Δ=0, Pocock Δ=.5)
    Wt(f64),
    /// Haybittle–Peto: fixed interim bound, final bound adjusted
    Haybittle(f64),
    /// error-spending families
    ErrObf,
    ErrPocock,
    Kdemets(f64),
    Hsdecani(f64),
}

impl Family {
    pub fn parse(s: &str) -> Result<Family, String> {
        let s = s.trim().to_lowercase();
        let (name, arg) = match s.find('(') {
            Some(i) => (s[..i].trim().to_string(), s[i + 1..].trim_end_matches(')').trim().parse::<f64>().ok()),
            None => (s.clone(), None),
        };
        Ok(match name.as_str() {
            "" | "none" => Family::None,
            "obfleming" | "obf" => Family::Wt(0.0),
            "pocock" => Family::Wt(0.5),
            "wtsiatis" => Family::Wt(arg.ok_or_else(|| rc_err(198, "wtsiatis(#) requires the shape parameter"))?),
            "haybittle" | "peto" | "hpeto" => Family::Haybittle(arg.unwrap_or(3.0)),
            "errobfleming" | "errobf" => Family::ErrObf,
            "errpocock" => Family::ErrPocock,
            "kdemets" | "power" => Family::Kdemets(arg.unwrap_or(1.0)),
            "hsdecani" => Family::Hsdecani(arg.unwrap_or(-4.0)),
            other => return Err(rc_err(198, format!("boundary {other} not allowed"))),
        })
    }

    pub fn name(&self) -> String {
        match self {
            Family::None => "none".into(),
            Family::Wt(d) if *d == 0.0 => "O'Brien-Fleming".into(),
            Family::Wt(d) if *d == 0.5 => "Pocock".into(),
            Family::Wt(d) => format!("Wang-Tsiatis (Δ = {d})"),
            Family::Haybittle(c) => format!("Haybittle-Peto (interim z = {c})"),
            Family::ErrObf => "O'Brien-Fleming-type error spending".into(),
            Family::ErrPocock => "Pocock-type error spending".into(),
            Family::Kdemets(r) => format!("Kim-DeMets error spending (rho = {r})"),
            Family::Hsdecani(g) => format!("Hwang-Shih-DeCani error spending (gamma = {g})"),
        }
    }

    fn spending(&self) -> bool {
        matches!(self, Family::ErrObf | Family::ErrPocock | Family::Kdemets(_) | Family::Hsdecani(_))
    }

    /// Cumulative error spent at information `t` out of `total`. For the
    /// classical families used as futility rules their spending analogue.
    pub fn spend(&self, t: f64, total: f64) -> f64 {
        if t >= 1.0 {
            return total;
        }
        match self {
            Family::Wt(d) if *d >= 0.5 => total * (1.0 + (std::f64::consts::E - 1.0) * t).ln(),
            Family::ErrObf | Family::Wt(_) | Family::Haybittle(_) => 2.0 - 2.0 * dist::normal(dist::invnormal(1.0 - total / 2.0) / t.sqrt()),
            Family::ErrPocock => total * (1.0 + (std::f64::consts::E - 1.0) * t).ln(),
            Family::Kdemets(r) => total * t.powf(*r),
            Family::Hsdecani(gm) => {
                if gm.abs() < 1e-12 {
                    total * t
                } else {
                    total * (1.0 - (-gm * t).exp()) / (1.0 - (-gm).exp())
                }
            }
            Family::None => 0.0,
        }
    }
}

/// A computed design on the z scale.
#[derive(Clone, Debug)]
pub struct Design {
    pub t: Vec<f64>,
    pub eff: Vec<f64>,
    /// futility bounds (NaN when none at a look)
    pub fut: Vec<f64>,
    pub two_sided: bool,
    /// drift: E[Z at t = 1] under the alternative
    pub theta: f64,
    /// cumulative type I error spent (both sides for two-sided designs)
    pub alpha_cum: Vec<f64>,
    /// cumulative type II error spent
    pub beta_cum: Vec<f64>,
    /// stopping probabilities per look under H0 and H1
    pub stop0: Vec<f64>,
    pub stop1: Vec<f64>,
    pub power: f64,
}

/// Continuation region at a look given efficacy bound `c` and futility `f`.
fn continuation(c: f64, f: f64, two: bool) -> Vec<(f64, f64)> {
    if two {
        if f.is_nan() || f <= 0.0 {
            vec![(-c, c)]
        } else if f >= c {
            vec![]
        } else {
            vec![(-c, -f), (f, c)]
        }
    } else {
        let lo = if f.is_nan() { f64::NEG_INFINITY } else { f };
        if lo >= c {
            vec![]
        } else {
            vec![(lo, c)]
        }
    }
}

/// Upper-crossing probability at a look (plus the lower one if two-sided).
fn cross(st: &State, tk: f64, theta: f64, c: f64, two: bool) -> f64 {
    let up = st.prob(tk, theta, c, f64::INFINITY);
    if two {
        up + st.prob(tk, theta, f64::NEG_INFINITY, -c)
    } else {
        up
    }
}

/// Futility-stopping probability at a look.
fn fut_prob(st: &State, tk: f64, theta: f64, f: f64, two: bool) -> f64 {
    if f.is_nan() {
        return 0.0;
    }
    if two {
        if f <= 0.0 {
            0.0
        } else {
            st.prob(tk, theta, -f, f)
        }
    } else {
        st.prob(tk, theta, f64::NEG_INFINITY, f)
    }
}

fn bisect(mut lo: f64, mut hi: f64, iters: usize, up: impl Fn(f64) -> bool) -> f64 {
    // `up(x)` true means the root is below x
    for _ in 0..iters {
        let mid = 0.5 * (lo + hi);
        if up(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Efficacy bounds under H0 without futility (classical families).
fn classical_bounds(fam: &Family, t: &[f64], alpha: f64, two: bool) -> Vec<f64> {
    let k = t.len();
    let shape = |cc: f64| -> Vec<f64> {
        match fam {
            Family::Wt(d) => t.iter().map(|ti| cc * ti.powf(d - 0.5)).collect(),
            Family::Haybittle(c) => (0..k).map(|i| if i + 1 < k { *c } else { cc }).collect(),
            _ => vec![cc; k],
        }
    };
    let total = |cc: f64| -> f64 {
        let b = shape(cc);
        let mut st = State::start();
        let mut tot = 0.0;
        for i in 0..k {
            tot += cross(&st, t[i], 0.0, b[i], two);
            if i + 1 < k {
                st = st.advance(t[i], 0.0, &continuation(b[i], f64::NAN, two));
            }
        }
        tot
    };
    let cc = bisect(0.0, 12.0, 60, |x| total(x) <= alpha);
    shape(cc)
}

/// Spend up to `inc` at a look: the bound c with crossing probability `inc`.
fn spend_bound(st: &State, tk: f64, inc: f64, two: bool) -> f64 {
    if inc <= 1e-15 {
        return 12.0f64.min(40.0);
    }
    bisect(-12.0, 40.0, 70, |x| cross(st, tk, 0.0, x, two) <= inc)
}

/// Compute a design. `theta` is solved for the target power unless given.
pub fn compute(t: &[f64], eff: &Family, fut: &Family, alpha: f64, beta: f64, two: bool, binding: bool) -> Result<Design, String> {
    let k = t.len();
    if k == 0 || t.windows(2).any(|w| w[1] <= w[0]) || t[0] <= 0.0 || (t[k - 1] - 1.0).abs() > 1e-9 {
        return Err(rc_err(198, "information fractions must increase to 1"));
    }
    if *eff == Family::None {
        return Err(rc_err(198, "an efficacy boundary is required"));
    }
    let has_fut = *fut != Family::None;
    // efficacy bounds that do not depend on θ
    let fixed_eff: Option<Vec<f64>> = if !eff.spending() {
        Some(classical_bounds(eff, t, alpha, two))
    } else if !(has_fut && binding) {
        let mut b = Vec::with_capacity(k);
        let mut st = State::start();
        let mut prev = 0.0;
        for i in 0..k {
            let cum = eff.spend(t[i], alpha);
            let c = spend_bound(&st, t[i], (cum - prev).max(0.0), two);
            prev = cum;
            b.push(c);
            if i + 1 < k {
                st = st.advance(t[i], 0.0, &continuation(c, f64::NAN, two));
            }
        }
        Some(b)
    } else {
        None
    };
    // a full pass for drift θ: bounds, error spent and stopping probabilities
    let pass = |theta: f64| -> Design {
        let mut st0 = State::start();
        let mut st1 = State::start();
        let mut effb = Vec::with_capacity(k);
        let mut futb = Vec::with_capacity(k);
        let (mut acum, mut bcum) = (Vec::new(), Vec::new());
        let (mut s0, mut s1) = (Vec::new(), Vec::new());
        let (mut a_tot, mut b_tot, mut pw) = (0.0, 0.0, 0.0);
        let mut prev_a = 0.0;
        for i in 0..k {
            let c = match &fixed_eff {
                Some(b) => b[i],
                None => {
                    let cum = eff.spend(t[i], alpha);
                    let c = spend_bound(&st0, t[i], (cum - prev_a).max(0.0), two);
                    prev_a = cum;
                    c
                }
            };
            let mut f = f64::NAN;
            if has_fut {
                if i + 1 == k {
                    f = c;
                } else {
                    let inc = (fut.spend(t[i], beta) - if i == 0 { 0.0 } else { fut.spend(t[i - 1], beta) }).max(0.0);
                    if inc > 1e-14 {
                        f = bisect(-12.0, 40.0, 70, |x| fut_prob(&st1, t[i], theta, x, two) >= inc);
                        if f > c {
                            f = c;
                        }
                        if two && f < 0.0 {
                            f = f64::NAN;
                        }
                    }
                }
            }
            let a_k = cross(&st0, t[i], 0.0, c, two);
            let fut0 = if has_fut && binding { fut_prob(&st0, t[i], 0.0, f, two) } else { 0.0 };
            let up1 = st1.prob(t[i], theta, c, f64::INFINITY);
            let low1 = if two { st1.prob(t[i], theta, f64::NEG_INFINITY, -c) } else { 0.0 };
            let fut1 = if has_fut { fut_prob(&st1, t[i], theta, f, two) } else { 0.0 };
            a_tot += a_k;
            pw += up1;
            b_tot += fut1;
            acum.push(a_tot);
            bcum.push(b_tot);
            s0.push(a_k + fut0);
            s1.push(up1 + low1 + fut1);
            effb.push(c);
            futb.push(f);
            if i + 1 < k {
                let r0 = if binding { continuation(c, f, two) } else { continuation(c, f64::NAN, two) };
                st0 = st0.advance(t[i], 0.0, &r0);
                st1 = st1.advance(t[i], theta, &continuation(c, f, two));
            }
        }
        // the last look ends the trial
        let rest0: f64 = s0[..k - 1].iter().sum();
        let rest1: f64 = s1[..k - 1].iter().sum();
        s0[k - 1] = 1.0 - rest0;
        s1[k - 1] = 1.0 - rest1;
        if !has_fut {
            bcum = acum.iter().enumerate().map(|(i, _)| 0.0 * i as f64).collect();
        }
        Design { t: t.to_vec(), eff: effb, fut: futb, two_sided: two, theta, alpha_cum: acum, beta_cum: bcum, stop0: s0, stop1: s1, power: pw }
    };
    let target = 1.0 - beta;
    let z_fix = zcrit(alpha, !two) + dist::invnormal(1.0 - beta);
    let theta = bisect(0.0, z_fix * 3.0 + 5.0, 50, |th| pass(th).power >= target);
    let mut d = pass(theta);
    if has_fut {
        d.beta_cum = d.beta_cum.iter().map(|b| b.min(beta)).collect();
        if let Some(l) = d.beta_cum.last_mut() {
            *l = 1.0 - d.power;
        }
    }
    Ok(d)
}

impl Design {
    /// Ratio of the maximum information to that of the fixed-sample design.
    pub fn inflation(&self, alpha: f64, beta: f64) -> f64 {
        let z = zcrit(alpha, !self.two_sided) + dist::invnormal(1.0 - beta);
        (self.theta / z).powi(2)
    }
}

// ---------------------------------------------------------------------------
// commands
// ---------------------------------------------------------------------------

/// Settings shared by gsbounds and gsdesign.
pub struct GsSpec {
    pub t: Vec<f64>,
    pub eff: Family,
    pub fut: Family,
    pub alpha: f64,
    pub beta: f64,
    pub two: bool,
    pub binding: bool,
}

pub fn gs_spec(cmd: &Command) -> Result<GsSpec, String> {
    let t: Vec<f64> = match cmd.opt("information", 4) {
        Some(s) => {
            let v = crate::margins::numlist(s.trim())?;
            let last = *v.last().ok_or_else(|| rc_err(198, "information() empty"))?;
            v.iter().map(|x| x / last).collect()
        }
        None => {
            let k = crate::util::opt_num(cmd, "nlooks", 2)?.unwrap_or(2.0) as usize;
            if k < 1 {
                return Err(rc_err(198, "nlooks() must be at least 1"));
            }
            (1..=k).map(|i| i as f64 / k as f64).collect()
        }
    };
    let eff = Family::parse(cmd.opt("efficacy", 3).unwrap_or("obfleming"))?;
    let fut = Family::parse(cmd.opt("futility", 3).unwrap_or("none"))?;
    let alpha = crate::util::opt_num(cmd, "alpha", 1)?.unwrap_or(0.05);
    let beta = match (crate::util::opt_num(cmd, "power", 1)?, crate::util::opt_num(cmd, "beta", 1)?) {
        (Some(p), _) => 1.0 - p,
        (None, Some(b)) => b,
        _ => 0.2,
    };
    if !(alpha > 0.0 && alpha < 1.0 && beta > 0.0 && beta < 1.0) {
        return Err(rc_err(198, "alpha() and power() must be between 0 and 1"));
    }
    Ok(GsSpec { t, eff, fut, alpha, beta, two: !cmd.has_opt("onesided", 3), binding: cmd.has_opt("binding", 4) && !cmd.has_opt("nonbinding", 4) })
}

fn fmtz(v: f64) -> String {
    if v.is_nan() {
        ".".into()
    } else if v.is_infinite() {
        if v > 0.0 { "+inf" } else { "-inf" }.into()
    } else {
        format!("{v:.4}")
    }
}

/// Fixed-sample sizes the group-sequential design scales up.
pub struct Fixed {
    pub title: String,
    pub test: String,
    /// (name, value) of the fixed-design sizes, e.g. N, N1, N2 or E
    pub sizes: Vec<(String, f64)>,
    pub params: Vec<(String, f64)>,
}

/// Run the design and display it (with sample sizes when `fixed` is given).
pub fn run(st: &mut Store, _cmd: &Command, spec: &GsSpec, fixed: Option<Fixed>) -> Result<String, String> {
    let d = compute(&spec.t, &spec.eff, &spec.fut, spec.alpha, spec.beta, spec.two, spec.binding)?;
    let infl = d.inflation(spec.alpha, spec.beta);
    let k = d.t.len();
    let mut out = vec![String::new()];
    match &fixed {
        Some(f) => out.push(format!("Group sequential design for {}", f.title)),
        None => out.push("Group sequential boundaries".into()),
    }
    if let Some(f) = &fixed {
        if !f.test.is_empty() {
            out.push(f.test.clone());
        }
    }
    out.push(format!("Efficacy: {}", spec.eff.name()));
    out.push(format!("Futility: {}{}", spec.fut.name(), if spec.fut != Family::None { if spec.binding { " (binding)" } else { " (nonbinding)" } } else { "" }));
    out.push(String::new());
    out.push("Study parameters:".into());
    out.push(String::new());
    out.push(format!("{:>16} = {:>10.4}", "alpha", spec.alpha));
    out.push(format!("{:>16} = {:>10.4}", "power", 1.0 - spec.beta));
    out.push(format!("{:>16} = {:>10}", "looks", k));
    out.push(format!("{:>16} = {:>10}", "test", if spec.two { "two-sided" } else { "one-sided" }));
    if let Some(f) = &fixed {
        for (n, v) in &f.params {
            out.push(format!("{:>16} = {:>10.4}", n, v));
        }
    }
    out.push(String::new());
    out.push(format!("{:>16} = {:>10.4}", "drift (theta)", d.theta));
    out.push(format!("{:>16} = {:>10.4}", "inflation factor", infl));
    out.push(format!("{:>16} = {:>10.4}", "achieved power", d.power));
    let mut r = Results::new();
    let mut nmax: Vec<(String, f64)> = Vec::new();
    if let Some(f) = &fixed {
        out.push(String::new());
        for (n, v) in &f.sizes {
            let m = (v * infl - 1e-9).ceil();
            out.push(format!("{:>16} = {:>10}  (fixed design {})", format!("{n} max"), m, (v - 1e-9).ceil()));
            r.set(&format!("{n}_fixed"), (v - 1e-9).ceil());
            r.set(&format!("{n}_max"), m);
            nmax.push((n.clone(), m));
        }
        // expected sizes
        let en0: f64 = d.t.iter().zip(&d.stop0).map(|(t, p)| t * p).sum();
        let en1: f64 = d.t.iter().zip(&d.stop1).map(|(t, p)| t * p).sum();
        if let Some((n, m)) = nmax.first() {
            out.push(format!("{:>16} = {:>10.2}", format!("E({n}) under H0"), m * en0));
            out.push(format!("{:>16} = {:>10.2}", format!("E({n}) under Ha"), m * en1));
            r.set(&format!("{n}_e0"), m * en0);
            r.set(&format!("{n}_e1"), m * en1);
        }
    }
    out.push(String::new());
    let mut head = format!("{:>5} {:>10}", "Look", "Info frac");
    for (n, _) in &nmax {
        head.push_str(&format!(" {:>8}", n));
    }
    head.push_str(&format!(" {:>10} {:>10} {:>10} {:>10}", "Efficacy", "Futility", "Cum alpha", "Cum beta"));
    let width = head.len();
    out.push("-".repeat(width));
    out.push(head);
    out.push("-".repeat(width));
    let mut bounds = Matrix::zeros(k, 6 + nmax.len());
    for i in 0..k {
        let mut line = format!("{:>5} {:>10.4}", i + 1, d.t[i]);
        for (_, m) in &nmax {
            line.push_str(&format!(" {:>8}", (m * d.t[i] - 1e-9).ceil()));
        }
        line.push_str(&format!(" {:>10} {:>10} {:>10.5} {:>10.5}", fmtz(d.eff[i]), fmtz(d.fut[i]), d.alpha_cum[i], d.beta_cum[i]));
        out.push(line);
        let row = [i as f64 + 1.0, d.t[i], d.eff[i], if d.fut[i].is_nan() { ferrostat_core::missing::SYSMISS } else { d.fut[i] }, d.alpha_cum[i], d.beta_cum[i]];
        for (j, v) in row.iter().enumerate() {
            bounds[(i, j)] = *v;
        }
        for (j, (_, m)) in nmax.iter().enumerate() {
            bounds[(i, 6 + j)] = (m * d.t[i] - 1e-9).ceil();
        }
    }
    out.push("-".repeat(width));
    if spec.two {
        out.push("Two-sided design: reject H0 when |z| exceeds the efficacy bound.".into());
    }
    let mut cn: Vec<String> = ["look", "info_frac", "efficacy", "futility", "alpha_cum", "beta_cum"].iter().map(|s| s.to_string()).collect();
    cn.extend(nmax.iter().map(|(n, _)| n.clone()));
    bounds.colnames = cn;
    bounds.rownames = (1..=k).map(|i| format!("look{i}")).collect();
    r.set_mat("bounds", bounds);
    r.set("nlooks", k as f64);
    r.set("alpha", spec.alpha);
    r.set("beta", spec.beta);
    r.set("power", 1.0 - spec.beta);
    r.set("power_achieved", d.power);
    r.set("drift", d.theta);
    r.set("info_ratio", infl);
    r.set("onesided", if spec.two { 0.0 } else { 1.0 });
    r.set("binding", if spec.binding { 1.0 } else { 0.0 });
    r.set_str("efficacy", spec.eff.name());
    r.set_str("futility", spec.fut.name());
    if let Some(f) = &fixed {
        for (n, v) in &f.params {
            r.set(n, *v);
        }
    }
    st.r = r;
    Ok(out.join("\n"))
}

pub struct Gsbounds;

impl CommandHandler for Gsbounds {
    fn name(&self) -> &'static str {
        "gsbounds"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let spec = gs_spec(cmd)?;
        let out = run(st, cmd, &spec, None)?;
        st.r.set_str("cmd", "gsbounds");
        Ok(out)
    }
}

pub struct Gsdesign;

/// Fixed-sample size of a built-in power method at the design's α and power.
fn fixed_for(cmd: &Command, method: &str, rest: &str, spec: &GsSpec) -> Result<Fixed, String> {
    let canon = super::methods::canonical(method).ok_or_else(|| rc_err(199, format!("gsdesign method {method} not found")))?;
    if !matches!(canon, "onemean" | "twomeans" | "oneproportion" | "twoproportions" | "logrank") {
        return Err(rc_err(198, format!("gsdesign {canon} not available")));
    }
    let pos = positional(rest)?;
    let (mut def, mut lists) = super::methods::setup(canon, cmd, &pos)?;
    super::common_lists(cmd, &def, &mut lists, false)?;
    lists.retain(|(k, _)| !matches!(k.as_str(), "power" | "alpha" | "_onesided" | "N" | "N1" | "N2"));
    lists.push(("alpha".into(), vec![spec.alpha]));
    lists.push(("power".into(), vec![1.0 - spec.beta]));
    lists.push(("_nfrac".into(), vec![1.0]));
    lists.push(("_z".into(), vec![1.0]));
    if !spec.two {
        lists.push(("_onesided".into(), vec![1.0]));
    }
    let rows = grid(&lists, false)?;
    if rows.len() != 1 {
        return Err(rc_err(198, "gsdesign does not allow lists of parameter values"));
    }
    let mut r: Row = rows.into_iter().next().unwrap();
    super::methods::prepare(&def, &mut r)?;
    if def.target.is_some_and(|t| !has(&r, t) && !super::target_implied(&r, t)) && !matches!(canon, "logrank") {
        return Err(rc_err(198, "the alternative value is required"));
    }
    super::solve_n(&def, &mut r, 1.0 - spec.beta, None)?;
    if canon.contains("mean") {
        def.test = "z test".into();
    }
    let mut sizes = vec![];
    if canon == "logrank" {
        let (a, b) = super::eff_sizes(&def, &r);
        let e = def.extra.map(|f| f(&r, a, b)).unwrap_or_default();
        if let Some((_, ev)) = e.iter().find(|(k, _)| k == "E") {
            sizes.push(("E".to_string(), *ev));
        }
    }
    if def.two {
        sizes.push(("N".into(), g(&r, "N1") + g(&r, "N2")));
        sizes.push(("N1".into(), g(&r, "N1")));
        sizes.push(("N2".into(), g(&r, "N2")));
    } else {
        sizes.push(("N".into(), g(&r, "N")));
    }
    let mut params = vec![("delta".to_string(), (def.delta)(&r))];
    for k in &def.show {
        if has(&r, k) {
            params.push((k.clone(), g(&r, k)));
        }
    }
    Ok(Fixed { title: def.title.clone(), test: def.test.clone(), sizes, params })
}

impl CommandHandler for Gsdesign {
    fn name(&self) -> &'static str {
        "gsdesign"
    }
    fn execute(&self, _ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let spec = gs_spec(cmd)?;
        let main = cmd.main.trim();
        let (sub, rest) = main.split_once(char::is_whitespace).unwrap_or((main, ""));
        let fixed = if sub.is_empty() { None } else { Some(fixed_for(cmd, &sub.to_lowercase(), rest, &spec)?) };
        let out = run(st, cmd, &spec, fixed)?;
        st.r.set_str("cmd", "gsdesign");
        if !sub.is_empty() {
            st.r.set_str("method", super::methods::canonical(sub).unwrap_or(sub));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pocock_two_looks() {
        let b = classical_bounds(&Family::Wt(0.5), &[0.5, 1.0], 0.05, true);
        assert!((b[0] - 2.178).abs() < 2e-3, "{b:?}");
    }
}
