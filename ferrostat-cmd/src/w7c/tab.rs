//! Coefficient tables for the latent-variable estimators: equation and
//! section headers, constrained parameters and ancillary rows without tests.

use crate::dist;
use crate::format::fmt_g;
use crate::matrix::Matrix;

pub enum Row {
    /// section or equation header (left aligned)
    Head(String),
    /// label, estimate, std. err. (None: constrained), show z test
    Coef(String, f64, Option<f64>, bool),
    /// estimate shown without std. err. (e.g. derived)
    Rule,
    /// free text in the value columns
    Note(String),
}

fn cell(v: f64) -> String {
    fmt_g(v, 8)
}

pub fn ab(s: &str, w: usize) -> String {
    crate::summarize::abbrev(s, w)
}

/// Render the table; also stashes r(table).
pub fn table(vcetype: &str, level: f64, coef_title: &str, rows: &[Row]) -> Vec<String> {
    let lw = 12usize;
    let mut out = Vec::new();
    let bar = "-".repeat(lw + 67);
    out.push(bar.clone());
    if !vcetype.is_empty() {
        out.push(format!("{:>lw$} | {:>w$}", "", vcetype, w = 23 + vcetype.len() / 2));
    }
    out.push(format!("{:>lw$} | {:>11}  std. err.      z    P>|z|     [{}% conf. interval]", "", coef_title, fmt_g(level, 4)));
    out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(65)));
    let crit = dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0);
    let mut tabm: Vec<[f64; 6]> = Vec::new();
    let mut tabn: Vec<String> = Vec::new();
    for r in rows {
        match r {
            Row::Head(h) => out.push(format!("{:<lw$} |", ab(h, lw))),
            Row::Rule => out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(65))),
            Row::Note(t) => out.push(format!("{:>lw$} |  {}", "", t)),
            Row::Coef(l, b, se, test) => match se {
                None => {
                    out.push(format!("{:>lw$} | {:>10}  (constrained)", ab(l, lw), cell(*b)));
                    tabm.push([*b, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN]);
                    tabn.push(l.clone());
                }
                Some(se) if !(*se > 0.0) => {
                    out.push(format!("{:>lw$} | {:>10}  {:>9}", ab(l, lw), cell(*b), "."));
                    tabm.push([*b, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN]);
                    tabn.push(l.clone());
                }
                Some(se) => {
                    let (lo, hi) = (b - crit * se, b + crit * se);
                    if *test {
                        let z = b / se;
                        let p = 2.0 * (1.0 - dist::normal(z.abs()));
                        out.push(format!("{:>lw$} | {:>10}  {:>9}  {:>7.2}   {:.3}    {:>9}   {:>9}", ab(l, lw), cell(*b), cell(*se), z, p, cell(lo), cell(hi)));
                        tabm.push([*b, *se, z, p, lo, hi]);
                    } else {
                        out.push(format!("{:>lw$} | {:>10}  {:>9}{:22}{:>9}   {:>9}", ab(l, lw), cell(*b), cell(*se), "", cell(lo), cell(hi)));
                        tabm.push([*b, *se, f64::NAN, f64::NAN, lo, hi]);
                    }
                    tabn.push(l.clone());
                }
            },
        }
    }
    out.push(bar);
    let mut m = Matrix::zeros(6, tabm.len());
    for (j, r) in tabm.iter().enumerate() {
        for i in 0..6 {
            m[(i, j)] = if r[i].is_nan() { ferrostat_core::missing::SYSMISS } else { r[i] };
        }
    }
    m.rownames = ["b", "se", "z", "pvalue", "ll", "ul"].iter().map(|s| s.to_string()).collect();
    m.colnames = tabn;
    crate::est::stash_table(m, level);
    out
}

/// Header block: title on the left, statistics on the right.
pub fn header(title: &str, stats: &[(String, String)]) -> Vec<String> {
    let mut out = Vec::new();
    for (i, (k, v)) in stats.iter().enumerate() {
        let right = format!("{:<17} = {:>10}", k, v);
        if i == 0 {
            out.push(format!("{:<50}{}", title, right));
        } else {
            out.push(format!("{:<50}{}", "", right));
        }
    }
    if stats.is_empty() {
        out.push(title.to_string());
    }
    out
}

pub fn n_str(n: f64) -> String {
    crate::util::commas(n.round() as usize)
}

pub fn g(v: f64) -> String {
    fmt_g(v, 9)
}

/// Move a stashed coefficient table into `r` (so postestimation commands
/// that display tables keep their own r() results).
pub fn keep_table(r: &mut crate::store::Results) {
    if let Some((m, l)) = crate::est::take_table() {
        r.set_mat("table", m);
        r.set("level", l);
    }
}
