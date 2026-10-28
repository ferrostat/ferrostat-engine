//! Pairwise comparisons: `pwcompare` (of balanced marginal linear
//! predictions after estimation), `margins ..., pwcompare` (of predictive
//! margins) and `pwmean` (of group means with equal variances), with
//! multiple-comparison adjustments (Bonferroni, Šidák, Scheffé, Tukey).

use std::collections::BTreeMap;

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::DataSet;
use ferrostat_parser::parser::parse_line;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::summarize::abbrev;
use crate::util;

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![("pwcompare", Box::new(Pwcompare)), ("pwmean", Box::new(Pwmean))]
}

// ---------------------------------------------------------------------------
// Studentized range distribution
// ---------------------------------------------------------------------------

/// P(range of `cc` standard normals < w) (Copenhaver & Holland 1988).
fn wprob(w: f64, rr: f64, cc: f64) -> f64 {
    const XLEG: [f64; 6] = [0.981560634246719250690549090149, 0.904117256370474856678465866119, 0.769902674194304687036893833213, 0.587317954286617447296702418941, 0.367831498998180193752691536644, 0.125233408511468915472441369464];
    const ALEG: [f64; 6] = [0.047175336386511827194615961485, 0.106939325995318430960254718194, 0.160078328543346226334652529543, 0.203167426723065921749064455810, 0.233492536538354808760849898925, 0.249147045813402785000562436043];
    let (c1, c2, c3, bb) = (-30.0f64, -50.0f64, 60.0f64, 8.0f64);
    let qsqz = w * 0.5;
    if qsqz >= bb {
        return 1.0;
    }
    let mut pr_w = 2.0 * dist::normal(qsqz) - 1.0;
    pr_w = if pr_w >= (c2 / cc).exp() { pr_w.powf(cc) } else { 0.0 };
    let wincr = if w > 3.0 { 2.0 } else { 3.0 };
    let mut blb = qsqz;
    let binc = (bb - qsqz) / wincr;
    let mut bub = blb + binc;
    let mut einsum = 0.0;
    let cc1 = cc - 1.0;
    let mut wi = 1.0;
    while wi <= wincr {
        let mut elsum = 0.0;
        let a = 0.5 * (bub + blb);
        let b = 0.5 * (bub - blb);
        for jj in 1..=12usize {
            let (j, xx) = if 6 < jj { (12 - jj + 1, XLEG[12 - jj]) } else { (jj, -XLEG[jj - 1]) };
            let c = b * xx;
            let ac = a + c;
            let qexpo = ac * ac;
            if qexpo > c3 {
                break;
            }
            let pplus = 2.0 * dist::normal(ac);
            let pminus = 2.0 * dist::normal(ac - w);
            let mut rinsum = pplus * 0.5 - pminus * 0.5;
            if rinsum >= (c1 / cc1).exp() {
                rinsum = ALEG[j - 1] * (-(0.5 * qexpo)).exp() * rinsum.powf(cc1);
                elsum += rinsum;
            }
        }
        elsum *= (2.0 * b) * cc / (2.0 * std::f64::consts::PI).sqrt();
        einsum += elsum;
        blb = bub;
        bub += binc;
        wi += 1.0;
    }
    pr_w += einsum;
    if pr_w <= (c1 / rr).exp() {
        return 0.0;
    }
    pr_w = pr_w.powf(rr);
    pr_w.min(1.0)
}

/// Cumulative distribution of the studentized range with `cc` means and
/// `df` degrees of freedom (df = infinity for the normal case).
pub fn ptukey(q: f64, cc: f64, df: f64) -> f64 {
    const XLEGQ: [f64; 8] = [0.989400934991649932596154173450, 0.944575023073232576077988415535, 0.865631202387831743880467897712, 0.755404408355003033895101194847, 0.617876244402643748446671764049, 0.458016777657227386342419442984, 0.281603550779258913230460501460, 0.950125098376374401853193354250e-1];
    const ALEGQ: [f64; 8] = [0.271524594117540948517805724560e-1, 0.622535239386478928628438369944e-1, 0.951585116824927848099251076022e-1, 0.124628971255533872052476282192, 0.149595988816576732081501730547, 0.169156519395002538189312079030, 0.182603415044923588866763667969, 0.189450610455068496285396723208];
    let rr = 1.0;
    if q <= 0.0 {
        return 0.0;
    }
    if !q.is_finite() {
        return 1.0;
    }
    if !df.is_finite() || df > 25000.0 {
        return wprob(q, rr, cc);
    }
    let f2 = df * 0.5;
    let mut f2lf = f2 * df.ln() - df * std::f64::consts::LN_2 - dist::lngamma(f2);
    let f21 = f2 - 1.0;
    let ff4 = df * 0.25;
    let ulen: f64 = if df <= 100.0 {
        1.0
    } else if df <= 800.0 {
        0.5
    } else if df <= 5000.0 {
        0.25
    } else {
        0.125
    };
    f2lf += ulen.ln();
    let mut ans = 0.0;
    for i in 1..=50 {
        let mut otsum = 0.0;
        let twa1 = (2 * i - 1) as f64 * ulen;
        for jj in 1..=16usize {
            let (j, t1) = if 8 < jj {
                let j = jj - 8 - 1;
                (j, (f2lf + f21 * (twa1 + XLEGQ[j] * ulen).ln()) - (XLEGQ[j] * ulen + twa1) * ff4)
            } else {
                let j = jj - 1;
                (j, (f2lf + f21 * (twa1 - XLEGQ[j] * ulen).ln()) + (XLEGQ[j] * ulen - twa1) * ff4)
            };
            if t1 >= -30.0 {
                let qsqz = if 8 < jj { q * ((XLEGQ[j] * ulen + twa1) * 0.5).sqrt() } else { q * ((-(XLEGQ[j] * ulen) + twa1) * 0.5).sqrt() };
                let wprb = wprob(qsqz, rr, cc);
                otsum += wprb * ALEGQ[j] * t1.exp();
            }
        }
        if i as f64 * ulen >= 1.0 && otsum <= 1e-14 {
            break;
        }
        ans += otsum;
    }
    ans.min(1.0)
}

/// Quantile of the studentized range distribution.
pub fn qtukey(p: f64, cc: f64, df: f64) -> f64 {
    let (mut lo, mut hi) = (0.0, 1.0);
    while ptukey(hi, cc, df) < p && hi < 1e4 {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if ptukey(mid, cc, df) < p {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-12 * hi.max(1.0) {
            break;
        }
    }
    0.5 * (lo + hi)
}

// ---------------------------------------------------------------------------
// Multiple comparisons
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Method {
    None,
    Bonferroni,
    Sidak,
    Scheffe,
    Tukey,
}

impl Method {
    fn parse(cmd: &Command) -> Result<Method, String> {
        let Some(m) = cmd.opt("mcompare", 3) else { return Ok(Method::None) };
        let w = m.split(',').next().unwrap_or("").trim().to_lowercase();
        let is = |full: &str, min: usize| w.len() >= min && full.starts_with(w.as_str());
        Ok(if is("noadjust", 5) {
            Method::None
        } else if is("bonferroni", 3) {
            Method::Bonferroni
        } else if is("sidak", 3) {
            Method::Sidak
        } else if is("scheffe", 3) {
            Method::Scheffe
        } else if is("tukey", 3) {
            Method::Tukey
        } else {
            return Err(rc_err(198, format!("mcompare({w}) not allowed")));
        })
    }
    fn title(self) -> &'static str {
        match self {
            Method::None => "Unadjusted",
            Method::Bonferroni => "Bonferroni",
            Method::Sidak => "Sidak",
            Method::Scheffe => "Scheffe",
            Method::Tukey => "Tukey",
        }
    }
    fn key(self) -> &'static str {
        match self {
            Method::None => "noadjust",
            Method::Bonferroni => "bonferroni",
            Method::Sidak => "sidak",
            Method::Scheffe => "scheffe",
            Method::Tukey => "tukey",
        }
    }
}

/// Adjusted p-value and critical value for one family of comparisons:
/// `m` comparisons among `k` margins.
fn adjust(method: Method, t: f64, df: Option<f64>, m: usize, k: usize, level: f64) -> (f64, f64) {
    let alpha = 1.0 - level / 100.0;
    let p_raw = |t: f64| match df {
        Some(d) => 2.0 * dist::ttail(d, t.abs()),
        None => 2.0 * dist::normal(-t.abs()),
    };
    let crit_raw = |a: f64| match df {
        Some(d) => dist::invttail(d, a / 2.0),
        None => dist::invnormal(1.0 - a / 2.0),
    };
    let mf = m as f64;
    let q = (k as f64 - 1.0).max(1.0);
    match method {
        Method::None => (p_raw(t), crit_raw(alpha)),
        Method::Bonferroni => ((mf * p_raw(t)).min(1.0), crit_raw(alpha / mf)),
        Method::Sidak => (1.0 - (1.0 - p_raw(t)).powf(mf), crit_raw(1.0 - (1.0 - alpha).powf(1.0 / mf))),
        Method::Scheffe => match df {
            Some(d) => (dist::ftail(q, d, t * t / q), (q * dist::invftail(q, d, alpha)).sqrt()),
            None => (dist::chi2tail(q, t * t), dist::invchi2tail(q, alpha).sqrt()),
        },
        Method::Tukey => {
            let d = df.unwrap_or(f64::INFINITY);
            let p = 1.0 - ptukey(t.abs() * 2f64.sqrt(), k as f64, d);
            (p.clamp(0.0, 1.0), qtukey(1.0 - alpha, k as f64, d) / 2f64.sqrt())
        }
    }
}

// ---------------------------------------------------------------------------
// Shared core: margins with their variance, grouped in terms
// ---------------------------------------------------------------------------

pub(crate) struct Term {
    /// Header (e.g. `fertilizer`, `a#b`).
    pub head: String,
    /// Indices into the margins vector.
    pub cells: Vec<usize>,
}

pub(crate) struct Margins {
    /// Names like `2.fertilizer` and labels like `2`.
    pub names: Vec<String>,
    pub labels: Vec<String>,
    pub est: Vec<f64>,
    pub v: Matrix,
    pub terms: Vec<Term>,
}

pub(crate) struct Display<'a> {
    pub title: &'a str,
    pub preamble: Vec<String>,
    pub depvar: &'a str,
    pub margin_title: &'a str,
    pub se_title: Option<&'a str>,
}

/// Comparisons of a set of margins; returns the output and fills `r`.
pub(crate) fn compare(m: &Margins, df: Option<f64>, cmd: &Command, level: f64, disp: &Display, r: &mut Results) -> Result<String, String> {
    let method = Method::parse(cmd)?;
    let effects = cmd.has_opt("effects", 3);
    let pveffects = cmd.has_opt("pveffects", 3);
    let cieffects = !pveffects || effects || cmd.has_opt("cieffects", 3);
    let show_p = effects || pveffects;
    let show_ci = effects || cieffects && !pveffects;
    let cimargins = cmd.has_opt("cimargins", 3);
    let stat = if df.is_some() { "t" } else { "z" };
    // (term, i, j): margin i minus margin j
    let mut comps: Vec<(usize, usize, usize)> = Vec::new();
    for (ti, t) in m.terms.iter().enumerate() {
        for a in 0..t.cells.len() {
            for b in a + 1..t.cells.len() {
                comps.push((ti, t.cells[b], t.cells[a]));
            }
        }
    }
    let nc = comps.len();
    let k = m.est.len();
    let mut l = Matrix::zeros(nc, k);
    for (c, &(_, i, j)) in comps.iter().enumerate() {
        l[(c, i)] = 1.0;
        l[(c, j)] = -1.0;
    }
    let est: Vec<f64> = comps.iter().map(|&(_, i, j)| m.est[i] - m.est[j]).collect();
    let vvs = l.mul(&m.v)?.mul(&l.transpose())?;
    let cname = |i: usize, j: usize| -> String {
        let (ni, nj) = (&m.names[i], &m.names[j]);
        let pi: Vec<&str> = ni.split('#').collect();
        let pj: Vec<&str> = nj.split('#').collect();
        pi.iter()
            .zip(&pj)
            .map(|(a, b)| {
                let (la, va) = a.split_once('.').unwrap_or(("", a));
                let (lb, _) = b.split_once('.').unwrap_or(("", b));
                format!("{}vs{}.{}", la.trim_end_matches(['b', 'o', 'n']), lb.trim_end_matches(['b', 'o', 'n']), va)
            })
            .collect::<Vec<_>>()
            .join("#")
    };
    let clabel = |i: usize, j: usize| -> String {
        let (a, b) = (&m.labels[i], &m.labels[j]);
        if a.contains(' ') || a.contains('#') {
            format!("({}) vs ({})", a.replace('#', " "), b.replace('#', " "))
        } else {
            format!("{a} vs {b}")
        }
    };
    let names_vs: Vec<String> = comps.iter().map(|&(_, i, j)| cname(i, j)).collect();
    let labels_vs: Vec<String> = comps.iter().map(|&(_, i, j)| clabel(i, j)).collect();
    let lw = m.terms.iter().map(|t| t.head.chars().count() + 1).chain(labels_vs.iter().map(|s| s.chars().count() + 1)).max().unwrap_or(12).clamp(12, 32);
    let ab = |x: &str| abbrev(x, lw);
    let mut out = vec![String::new(), disp.title.to_string()];
    if !disp.preamble.first().is_some_and(|l| l.starts_with("Model VCE")) {
        out.push(String::new());
    }
    out.extend(disp.preamble.iter().cloned());
    if !disp.preamble.is_empty() {
        out.push(String::new());
    }
    let m_of = |ti: usize| -> usize {
        let c = m.terms[ti].cells.len();
        c * (c.saturating_sub(1)) / 2
    };
    if method != Method::None {
        out.push("-".repeat(lw + 15));
        out.push(format!("{:>lw$} |    Number of", ""));
        out.push(format!("{:>lw$} |  comparisons", ""));
        out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(13)));
        for (ti, t) in m.terms.iter().enumerate() {
            out.push(format!("{:>lw$} |  {:>11}", ab(&t.head), m_of(ti)));
        }
        out.push("-".repeat(lw + 15));
        out.push(String::new());
    }
    let crit_unadj = match df {
        Some(d) => dist::invttail(d, (1.0 - level / 100.0) / 2.0),
        None => dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0),
    };
    let lvl = fmt_g(level, 4);
    // margins table
    if cimargins {
        let w = 48;
        out.push("-".repeat(lw + 2 + w));
        if let Some(se) = disp.se_title {
            out.push(format!("{:>lw$} | {:>22}", "", se));
        }
        out.push(format!("{:>lw$} | {:>10}   Std. err.     [{lvl}% conf. interval]", disp.depvar, disp.margin_title));
        out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(w)));
        for t in &m.terms {
            if m.terms.len() > 1 || t.head != "_cons" {
                out.push(format!("{:>lw$} |", ab(&t.head)));
            }
            for &i in &t.cells {
                let se = m.v[(i, i)].max(0.0).sqrt();
                out.push(format!("{:>lw$} |  {:>9}  {:>9}     {:>9}   {:>9}", ab(&format!("{} ", m.labels[i])), fmt_g(m.est[i], 8), fmt_g(se, 8), fmt_g(m.est[i] - crit_unadj * se, 8), fmt_g(m.est[i] + crit_unadj * se, 8)));
            }
        }
        out.push("-".repeat(lw + 2 + w));
        out.push(String::new());
    }
    // comparisons table
    let width = if show_p && show_ci {
        66
    } else if show_p {
        43
    } else {
        48
    };
    out.push("-".repeat(lw + 2 + width));
    let mt = method.title();
    if let Some(se) = disp.se_title {
        if show_p && show_ci {
            out.push(format!("{:>lw$} | {:>22} {:>16}  {:>21}", "", se, mt, mt));
        } else if show_p {
            out.push(format!("{:>lw$} | {:>22} {:>16}", "", se, mt));
        } else {
            out.push(format!("{:>lw$} | {:>22} {:>20}", "", se, mt));
        }
    } else if show_p && show_ci {
        out.push(format!("{:>lw$} | {:>39}  {:>21}", "", mt, mt));
    } else if show_p {
        out.push(format!("{:>lw$} | {:>39}", "", mt));
    } else {
        out.push(format!("{:>lw$} | {:>43}", "", mt));
    }
    let ptitle = format!("P>|{stat}|");
    if show_p && show_ci {
        out.push(format!("{:>lw$} |   Contrast   Std. err.  {:>6}    {ptitle}     [{lvl}% conf. interval]", disp.depvar, stat));
    } else if show_p {
        out.push(format!("{:>lw$} |   Contrast   Std. err.  {:>6}    {ptitle}", disp.depvar, stat));
    } else {
        out.push(format!("{:>lw$} |   Contrast   Std. err.     [{lvl}% conf. interval]", disp.depvar));
    }
    out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(width)));
    let mut table = Matrix::zeros(9, nc);
    let mut last = usize::MAX;
    for (c, &(ti, _, _)) in comps.iter().enumerate() {
        if ti != last {
            out.push(format!("{:>lw$} |", ab(&m.terms[ti].head)));
            last = ti;
        }
        let se = vvs[(c, c)].max(0.0).sqrt();
        let t = est[c] / se;
        let (p, crit) = adjust(method, t, df, m_of(ti), m.terms[ti].cells.len(), level);
        let (lo, hi) = (est[c] - crit * se, est[c] + crit * se);
        for (rix, x) in [est[c], se, t, p, lo, hi, df.unwrap_or(SYSMISS), crit, 0.0].iter().enumerate() {
            table[(rix, c)] = *x;
        }
        let lab = ab(&format!("{} ", labels_vs[c]));
        if show_p && show_ci {
            out.push(format!("{lab:>lw$} |  {:>9}  {:>9}  {:>7.2}   {:.3}     {:>9}   {:>9}", fmt_g(est[c], 8), fmt_g(se, 8), t, p, fmt_g(lo, 8), fmt_g(hi, 8)));
        } else if show_p {
            out.push(format!("{lab:>lw$} |  {:>9}  {:>9}  {:>7.2}   {:.3}", fmt_g(est[c], 8), fmt_g(se, 8), t, p));
        } else {
            out.push(format!("{lab:>lw$} |  {:>9}  {:>9}     {:>9}   {:>9}", fmt_g(est[c], 8), fmt_g(se, 8), fmt_g(lo, 8), fmt_g(hi, 8)));
        }
    }
    out.push("-".repeat(lw + 2 + width));
    // stored results
    let rown: Vec<String> = ["b", "se", stat, "pvalue", "ll", "ul", "df", "crit", "eform"].iter().map(|s| s.to_string()).collect();
    let mut mtab = Matrix::zeros(9, k);
    for i in 0..k {
        let se = m.v[(i, i)].max(0.0).sqrt();
        let t = m.est[i] / se;
        let p = match df {
            Some(d) => 2.0 * dist::ttail(d, t.abs()),
            None => 2.0 * dist::normal(-t.abs()),
        };
        for (rix, x) in [m.est[i], se, t, p, m.est[i] - crit_unadj * se, m.est[i] + crit_unadj * se, df.unwrap_or(SYSMISS), crit_unadj, 0.0].iter().enumerate() {
            mtab[(rix, i)] = if x.is_finite() { *x } else { SYSMISS };
        }
    }
    r.set_mat("b", Matrix::row_vector(&m.est).with_names(vec!["y1".into()], m.names.clone()));
    r.set_mat("V", m.v.clone().with_names(m.names.clone(), m.names.clone()));
    r.set_mat("b_vs", Matrix::row_vector(&est).with_names(vec!["y1".into()], names_vs.clone()));
    r.set_mat("V_vs", vvs.with_names(names_vs.clone(), names_vs.clone()));
    r.set_mat("table", mtab.with_names(rown.clone(), m.names.clone()));
    r.set_mat("table_vs", table.with_names(rown, names_vs.clone()));
    r.set_mat("L", l.with_names(names_vs, m.names.clone()));
    r.set("level", level);
    r.set("k_terms", m.terms.len() as f64);
    if let Some(d) = df {
        r.set("df_r", d);
    }
    r.set_str("mcmethod_vs", method.key());
    let ks: Vec<f64> = (0..m.terms.len()).map(|ti| m_of(ti) as f64).collect();
    r.set_mat("k_groups", Matrix::row_vector(&ks));
    Ok(out.join("\n"))
}

// ---------------------------------------------------------------------------
// pwcompare
// ---------------------------------------------------------------------------

fn factor_part(p: &str) -> Option<(f64, &str)> {
    let (l, v) = p.split_once('.')?;
    l.trim_end_matches(['b', 'o', 'n']).parse::<f64>().ok().map(|x| (x, v))
}

fn bare(c: &str) -> &str {
    c.rsplit(':').next().unwrap_or(c)
}

/// Levels of a factor variable named in the coefficients.
fn levels_of(b: &Matrix, var: &str) -> Vec<f64> {
    let mut lv: Vec<f64> = Vec::new();
    for c in &b.colnames {
        for p in bare(c).split('#') {
            if let Some((l, v)) = factor_part(p) {
                if v == var && !lv.contains(&l) {
                    lv.push(l);
                }
            }
        }
    }
    lv.sort_by(f64::total_cmp);
    lv
}

/// Value label (or the level itself) of a factor level.
fn level_label(ds: &DataSet, var: &str, l: f64) -> String {
    ds.get_var(var).and_then(|v| ds.value_labels.get(&v.value_label_name)).and_then(|vl| vl.lookup(l).map(|s| s.to_string())).unwrap_or_else(|| fmt_g(l, 12))
}

/// Balanced margin gradient at fixed levels of some factors.
fn cell_row(b: &Matrix, eq: &Option<String>, levels: &BTreeMap<String, Vec<f64>>, means: &BTreeMap<String, f64>, fixed: &BTreeMap<String, f64>) -> Vec<f64> {
    b.colnames
        .iter()
        .map(|c| {
            if let (Some(eq), Some((e, _))) = (eq, c.split_once(':')) {
                if e != eq {
                    return 0.0;
                }
            }
            let name = bare(c);
            if name == "_cons" {
                return 1.0;
            }
            if name.starts_with('/') {
                return 0.0;
            }
            let mut prod = 1.0;
            for p in name.split('#') {
                match factor_part(p) {
                    Some((l, v)) => match fixed.get(v) {
                        Some(&f) => prod *= if f == l { 1.0 } else { 0.0 },
                        None => prod /= levels.get(v).map_or(1, |x| x.len().max(1)) as f64,
                    },
                    None => {
                        let v = p.strip_prefix("o.").unwrap_or(p);
                        let v = v.strip_prefix("c.").unwrap_or(v);
                        prod *= means.get(v).copied().unwrap_or(0.0);
                    }
                }
            }
            prod
        })
        .collect()
}

pub struct Pwcompare;

impl CommandHandler for Pwcompare {
    fn name(&self) -> &'static str {
        "pwcompare"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let b = st.e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?.clone();
        let v = st.e.matrices.get("V").ok_or_else(|| rc_err(301, "last estimates not found"))?.clone();
        let level = util::level(cmd, st)?;
        let sample: Vec<usize> = st.e.sample.as_ref().map_or_else(Vec::new, |m| util::rows(m));
        let mut levels = BTreeMap::new();
        let mut means = BTreeMap::new();
        for c in &b.colnames {
            for p in bare(c).split('#') {
                match factor_part(p) {
                    Some((_, var)) => {
                        levels.entry(var.to_string()).or_insert_with(|| levels_of(&b, var));
                    }
                    None if p != "_cons" && !p.starts_with('/') => {
                        let var = p.strip_prefix("o.").unwrap_or(p);
                        let var = var.strip_prefix("c.").unwrap_or(var).to_string();
                        if !means.contains_key(&var) {
                            let m = util::col(ds, &var)
                                .map(|x| util::mean(&sample.iter().map(|&i| x[i]).filter(|x| !is_missing(*x)).collect::<Vec<_>>()))
                                .unwrap_or(0.0);
                            means.insert(var, m);
                        }
                    }
                    None => {}
                }
            }
        }
        let eq = b.colnames.first().and_then(|c| c.split_once(':')).map(|(e, _)| e.to_string());
        let mut names = Vec::new();
        let mut labels = Vec::new();
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut terms = Vec::new();
        for t in cmd.main.split_whitespace() {
            let vars: Vec<String> = t.split('#').map(|p| p.split_once('.').map_or(p, |x| x.1).to_string()).collect();
            let mut lvls = Vec::new();
            for vn in &vars {
                let lv = levels_of(&b, vn);
                if lv.is_empty() {
                    return Err(rc_err(111, format!("factor variable {vn} not found in the model")));
                }
                lvls.push(lv);
            }
            let mut cells = Vec::new();
            let mut combos: Vec<Vec<usize>> = vec![Vec::new()];
            for lv in &lvls {
                combos = combos.into_iter().flat_map(|p| (0..lv.len()).map(move |i| [p.clone(), vec![i]].concat())).collect();
            }
            for combo in combos {
                let mut fixed = BTreeMap::new();
                let mut nm = Vec::new();
                let mut lb = Vec::new();
                for (vi, &li) in combo.iter().enumerate() {
                    let l = lvls[vi][li];
                    fixed.insert(vars[vi].clone(), l);
                    nm.push(format!("{}.{}", fmt_g(l, 12), vars[vi]));
                    lb.push(level_label(ds, &vars[vi], l));
                }
                cells.push(rows.len());
                rows.push(cell_row(&b, &eq, &levels, &means, &fixed));
                names.push(nm.join("#"));
                labels.push(lb.join("#"));
            }
            terms.push(Term { head: vars.join("#"), cells });
        }
        if terms.is_empty() {
            return Err(rc_err(198, "pwcompare: marginlist required"));
        }
        let k = b.cols;
        let mut lm = Matrix::zeros(rows.len(), k);
        for (i, r) in rows.iter().enumerate() {
            for j in 0..k {
                lm[(i, j)] = r[j];
            }
        }
        let est: Vec<f64> = rows.iter().map(|r| r.iter().zip(&b.data).map(|(a, x)| a * x).sum()).collect();
        let vm = lm.mul(&v)?.mul(&lm.transpose())?;
        let df = crate::postest2::uses_t(&st.e);
        let m = Margins { names, labels, est, v: vm, terms };
        let mut r = Results::new();
        let disp = Display { title: "Pairwise comparisons of marginal linear predictions", preamble: vec!["Margins: asbalanced".into()], depvar: "", margin_title: "Margin", se_title: None };
        let out = compare(&m, df, cmd, level, &disp, &mut r)?;
        r.set_str("cmd", "pwcompare");
        r.set_mat("M", lm.with_names(m.names.clone(), b.colnames.clone()));
        if cmd.has_opt("post", 4) {
            let n = st.e.get("N");
            let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
            let sample = st.e.sample.clone();
            let mut e = r.clone();
            e.set_str("cmd", "pwcompare");
            e.set_str("est_cmd", ecmd);
            e.set_str("properties", "b V");
            if let Some(n) = n {
                e.set("N", n);
            }
            e.sample = sample;
            st.e = e;
        }
        st.r = r;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// margins ..., pwcompare
// ---------------------------------------------------------------------------

/// `margins ..., pwcompare[(suboptions)]`: pairwise comparisons of the margins.
pub fn margins_pw(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
    let pw_arg = cmd.opts.iter().find(|(n, _)| n.len() >= 2 && "pwcompare".starts_with(n.to_lowercase().as_str())).and_then(|(_, a)| a.clone()).unwrap_or_default();
    let raw = cmd.raw.trim();
    // rebuild the command without pwcompare(): options as written
    let comma = ferrostat_parser::parser::find_top_level(raw, ',').unwrap_or(raw.len());
    let keep: Vec<String> = cmd
        .opts
        .iter()
        .filter(|(n, _)| !(n.len() >= 2 && "pwcompare".starts_with(n.to_lowercase().as_str())) && !(n.len() >= 3 && "mcompare".starts_with(n.to_lowercase().as_str())))
        .map(|(n, a)| match a {
            Some(a) => format!("{n}({a})"),
            None => n.clone(),
        })
        .collect();
    let line = if keep.is_empty() { raw[..comma].to_string() } else { format!("{}, {}", &raw[..comma], keep.join(" ")) };
    let c = parse_line(&line).map_err(|e| rc_err(198, e.message))?;
    let text = crate::margins::Margins.execute(ds, st, &c)?;
    let rb = st.r.matrices.get("b").cloned().ok_or_else(|| rc_err(301, "margins results not found"))?;
    let rv = st.r.matrices.get("V").cloned().ok_or_else(|| rc_err(301, "margins results not found"))?;
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    let df = st.e.get("df_r").filter(|_| matches!(ecmd.as_str(), "regress" | "anova" | "reghdfe"));
    // group margins by term: names with the levels removed
    let mut terms: Vec<Term> = Vec::new();
    let mut labels = Vec::new();
    for (i, n) in rb.colnames.iter().enumerate() {
        let parts: Vec<&str> = n.split('#').collect();
        let head = parts.iter().map(|p| factor_part(p).map_or(p.to_string(), |(_, v)| v.to_string())).collect::<Vec<_>>().join("#");
        let lab = parts
            .iter()
            .map(|p| match factor_part(p) {
                Some((l, v)) => {
                    if v == "_at" {
                        fmt_g(l, 12)
                    } else {
                        level_label(ds, v, l)
                    }
                }
                None => p.to_string(),
            })
            .collect::<Vec<_>>()
            .join("#");
        labels.push(lab);
        match terms.iter_mut().find(|t| t.head == head) {
            Some(t) => t.cells.push(i),
            None => terms.push(Term { head, cells: vec![i] }),
        }
    }
    // the preamble of the margins output (between the title and the table)
    let lines: Vec<&str> = text.lines().collect();
    let mut preamble = Vec::new();
    let mut started = false;
    let mut title_kind = st.r.macros.get("title").cloned().unwrap_or_else(|| "Predictive margins".into());
    let mut right = String::new();
    for l in &lines {
        if l.starts_with("---") {
            break;
        }
        if !started {
            if !l.trim().is_empty() {
                started = true;
                right = l.trim_start().strip_prefix(title_kind.as_str()).unwrap_or("").trim().to_string();
            }
            continue;
        }
        preamble.push(l.to_string());
    }
    while preamble.first().is_some_and(|l| l.trim().is_empty()) {
        preamble.remove(0);
    }
    while preamble.last().is_some_and(|l| l.trim().is_empty()) {
        preamble.pop();
    }
    title_kind = title_kind.to_lowercase();
    let mut title = format!("Pairwise comparisons of {title_kind}");
    if !right.is_empty() {
        let w = 78usize.saturating_sub(title.len()).max(right.len() + 1);
        title = format!("{title}{right:>w$}");
    }
    let pc = parse_line(&format!("p , {pw_arg} {}", cmd.opt("mcompare", 3).map(|m| format!("mcompare({m})")).unwrap_or_default())).map_err(|e| rc_err(198, e.message))?;
    let level = util::level(cmd, st)?;
    let m = Margins { names: rb.colnames.clone(), labels, est: rb.data.clone(), v: rv, terms };
    let mut r = st.r.clone();
    let disp = Display { title: &title, preamble, depvar: "", margin_title: "Margin", se_title: Some("Delta-method") };
    let out = compare(&m, df, &pc, level, &disp, &mut r)?;
    st.r = r;
    Ok(out)
}

// ---------------------------------------------------------------------------
// pwmean
// ---------------------------------------------------------------------------

pub struct Pwmean;

impl CommandHandler for Pwmean {
    fn name(&self) -> &'static str {
        "pwmean"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = crate::varlist::expand(ds, &cmd.main)?;
        if vars.len() != 1 {
            return Err(rc_err(198, "pwmean varname, over(varlist)"));
        }
        let over = crate::varlist::expand(ds, cmd.opt("over", 2).ok_or_else(|| rc_err(198, "option over() required"))?)?;
        let level = util::level(cmd, st)?;
        let mut mask = util::touse(ds, st, cmd)?;
        let mut all = vars.clone();
        all.extend(over.iter().cloned());
        util::markout(ds, &mut mask, &all);
        let y = util::col(ds, &vars[0])?;
        let ocols: Vec<Vec<f64>> = over.iter().map(|o| util::col(ds, o)).collect::<Result<_, _>>()?;
        let mut groups: BTreeMap<Vec<u64>, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
        for i in 0..ds.n_obs {
            if !mask[i] {
                continue;
            }
            let key: Vec<f64> = ocols.iter().map(|c| c[i]).collect();
            let ord: Vec<u64> = key.iter().map(|x| { let b = x.to_bits(); if *x >= 0.0 { b ^ (1 << 63) } else { !b } }).collect();
            groups.entry(ord).or_insert_with(|| (key.clone(), Vec::new())).1.push(y[i]);
        }
        let k = groups.len();
        if k < 2 {
            return Err(rc_err(2000, "over() must identify at least two groups"));
        }
        let n: usize = groups.values().map(|g| g.1.len()).sum();
        let ssw: f64 = groups.values().map(|g| {
            let m = util::mean(&g.1);
            g.1.iter().map(|x| (x - m).powi(2)).sum::<f64>()
        }).sum();
        let df = (n - k) as f64;
        let s2 = ssw / df;
        let mut names = Vec::new();
        let mut labels = Vec::new();
        let mut est = Vec::new();
        let mut v = Matrix::zeros(k, k);
        for (gi, (key, ys)) in groups.values().enumerate() {
            names.push(key.iter().zip(&over).map(|(l, o)| format!("{}.{o}", fmt_g(*l, 12))).collect::<Vec<_>>().join("#"));
            labels.push(key.iter().zip(&over).map(|(l, o)| level_label(ds, o, *l)).collect::<Vec<_>>().join("#"));
            est.push(util::mean(ys));
            v[(gi, gi)] = s2 / ys.len() as f64;
        }
        let m = Margins { names, labels, est, v, terms: vec![Term { head: over.join("#"), cells: (0..k).collect() }] };
        let mut r = Results::new();
        let disp = Display { title: "Pairwise comparisons of means with equal variances", preamble: vec![format!("Over: {}", over.join(" "))], depvar: &vars[0], margin_title: "Mean", se_title: None };
        let mut c2 = cmd.clone();
        if cmd.has_opt("cimeans", 3) {
            c2.opts.push(("cimargins".into(), None));
        }
        let out = compare(&m, Some(df), &c2, level, &disp, &mut r)?;
        r.set_str("cmd", "pwmean");
        r.set("N", n as f64);
        st.r = r;
        Ok(out)
    }
}
