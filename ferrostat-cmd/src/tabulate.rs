//! tabulate (one- and two-way), tab1, tab2.

use ferrostat_core::missing::is_missing;
use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::util;
use crate::varlist;

/// Distinct categories of a variable among selected observations, in sort
/// order: (representative value, display text).
fn categories(ds: &DataSet, var: &Variable, mask: &[bool], labels: bool) -> Vec<(Value, String)> {
    let mut vals: Vec<Value> = (0..ds.n_obs).filter(|&i| mask[i]).map(|i| var.data[i].clone()).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vals.dedup_by(|a, b| a.partial_cmp(&b) == Some(std::cmp::Ordering::Equal));
    vals.into_iter()
        .map(|v| {
            let text = match &v {
                Value::Str(s) => s.clone(),
                x => {
                    let n = x.num();
                    let lab = if labels && !var.value_label_name.is_empty() {
                        ds.value_labels.get(&var.value_label_name).and_then(|vl| vl.lookup(n)).map(String::from)
                    } else {
                        None
                    };
                    lab.unwrap_or_else(|| fmt_g(n, 9))
                }
            };
            (v, text)
        })
        .collect()
}

fn cat_index(cats: &[(Value, String)], v: &Value) -> Option<usize> {
    cats.iter().position(|(c, _)| c.partial_cmp(v) == Some(std::cmp::Ordering::Equal))
}

fn trunc(s: &str, n: usize) -> String {
    crate::summarize::abbrev(s, n)
}

pub struct Tabulate;

impl CommandHandler for Tabulate {
    fn name(&self) -> &'static str {
        "tabulate"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        match vars.len() {
            1 => one_way(ds, st, cmd, &vars[0]),
            2 => two_way(ds, st, cmd, &vars[0], &vars[1]),
            0 => Err(rc_err(100, "varlist required")),
            _ => Err(rc_err(103, "too many variables specified")),
        }
    }
}

fn is_missing_cell(var: &Variable, i: usize) -> bool {
    match &var.data[i] {
        Value::Str(s) => s.is_empty(),
        x => is_missing(x.num()),
    }
}

fn one_way(ds: &mut DataSet, st: &mut Store, cmd: &Command, v: &str) -> Result<String, String> {
    let mut mask = util::touse(ds, st, cmd)?;
    let w = util::weights(ds, st, cmd, &mut mask, "fai")?;
    let var = ds.get_var(v).unwrap().clone();
    if !cmd.has_opt("missing", 1) {
        for i in 0..ds.n_obs {
            if mask[i] && is_missing_cell(&var, i) {
                mask[i] = false;
            }
        }
    }
    let labels = !cmd.has_opt("nolabel", 3);
    let cats = categories(ds, &var, &mask, labels);
    let mut freq = vec![0.0; cats.len()];
    for i in 0..ds.n_obs {
        if mask[i] {
            let k = cat_index(&cats, &var.data[i]).unwrap();
            freq[k] += w.as_ref().map_or(1.0, |w| w.w[i]);
        }
    }
    let total: f64 = freq.iter().sum();
    let mut order: Vec<usize> = (0..cats.len()).collect();
    if cmd.has_opt("sort", 4) {
        order.sort_by(|&a, &b| freq[b].total_cmp(&freq[a]));
    }
    let head = if var.label.is_empty() || cmd.has_opt("nolabel", 3) { v.to_string() } else { var.label.clone() };
    let mut out = Vec::new();
    if total == 0.0 {
        out.push("no observations".into());
    } else {
        out.push(format!("{:>11} |      Freq.     Percent        Cum.", trunc(&head, 11)));
        out.push("------------+-----------------------------------".into());
        let mut cum = 0.0;
        for &k in &order {
            let pct = 100.0 * freq[k] / total;
            cum += pct;
            out.push(format!("{:>11} | {:>10} {:>11.2} {:>11.2}", trunc(&cats[k].1, 11), fmt_g(freq[k], 10), pct, cum));
        }
        out.push("------------+-----------------------------------".into());
        out.push(format!("{:>11} | {:>10} {:>11.2}", "Total", fmt_g(total, 10), 100.0));
    }
    // generate() indicator variables
    if let Some(stub) = cmd.opt("generate", 2) {
        let stub = stub.trim();
        for (k, (cv, _)) in cats.iter().enumerate() {
            let name = format!("{stub}{}", k + 1);
            let mut nv = Variable::new(&name, VarType::Byte);
            nv.data = (0..ds.n_obs)
                .map(|i| {
                    if is_missing_cell(&var, i) && !cmd.has_opt("missing", 1) {
                        Value::Double(ferrostat_core::SYSMISS)
                    } else {
                        Value::Byte((var.data[i].partial_cmp(cv) == Some(std::cmp::Ordering::Equal)) as i8)
                    }
                })
                .collect();
            nv.label = format!("{v}=={}", cats[k].1);
            ds.add_variable(nv)?;
        }
    }
    let mut r = Results::new();
    r.set("N", total);
    r.set("r", cats.len() as f64);
    if let Some(m) = cmd.opt("matcell", 5) {
        st.matrices.insert(m.trim().to_string(), Matrix::col_vector(&freq));
    }
    if let Some(m) = cmd.opt("matrow", 5) {
        st.matrices.insert(m.trim().to_string(), Matrix::col_vector(&cats.iter().map(|c| c.0.num()).collect::<Vec<_>>()));
    }
    st.r = r;
    // `nofreq` on a one-way table leaves nothing to show
    if cmd.has_opt("nofreq", 5) {
        return Ok(String::new());
    }
    Ok(out.join("\n"))
}

/// Fisher's exact test for 2x2 tables: (two-sided p, one-sided p).
fn fisher_2x2(t: [[f64; 2]; 2]) -> (f64, f64) {
    let (a, b, c, d) = (t[0][0], t[0][1], t[1][0], t[1][1]);
    let r1 = a + b;
    let c1 = a + c;
    let n = a + b + c + d;
    let lo = (r1 + c1 - n).max(0.0) as i64;
    let hi = r1.min(c1) as i64;
    let p = |x: f64| (dist::lnfactorial(r1) + dist::lnfactorial(n - r1) + dist::lnfactorial(c1) + dist::lnfactorial(n - c1)
        - dist::lnfactorial(n) - dist::lnfactorial(x) - dist::lnfactorial(r1 - x) - dist::lnfactorial(c1 - x) - dist::lnfactorial(n - r1 - c1 + x))
        .exp();
    let p0 = p(a);
    let mut two = 0.0;
    let mut left = 0.0;
    let mut right = 0.0;
    for x in lo..=hi {
        let px = p(x as f64);
        if px <= p0 * (1.0 + 1e-7) {
            two += px;
        }
        if (x as f64) <= a {
            left += px;
        }
        if (x as f64) >= a {
            right += px;
        }
    }
    (two.min(1.0), left.min(right).min(1.0))
}

fn two_way(ds: &mut DataSet, st: &mut Store, cmd: &Command, v1: &str, v2: &str) -> Result<String, String> {
    let mut mask = util::touse(ds, st, cmd)?;
    let w = util::weights(ds, st, cmd, &mut mask, "fai")?;
    let a = ds.get_var(v1).unwrap().clone();
    let b = ds.get_var(v2).unwrap().clone();
    if !cmd.has_opt("missing", 1) {
        for i in 0..ds.n_obs {
            if mask[i] && (is_missing_cell(&a, i) || is_missing_cell(&b, i)) {
                mask[i] = false;
            }
        }
    }
    let labels = !cmd.has_opt("nolabel", 3);
    let rc = categories(ds, &a, &mask, labels);
    let cc = categories(ds, &b, &mask, labels);
    let (nr, nc) = (rc.len(), cc.len());
    let mut t = vec![vec![0.0; nc]; nr];
    for i in 0..ds.n_obs {
        if mask[i] {
            let r = cat_index(&rc, &a.data[i]).unwrap();
            let c = cat_index(&cc, &b.data[i]).unwrap();
            t[r][c] += w.as_ref().map_or(1.0, |w| w.w[i]);
        }
    }
    let rt: Vec<f64> = t.iter().map(|r| r.iter().sum()).collect();
    let ct: Vec<f64> = (0..nc).map(|c| t.iter().map(|r| r[c]).sum()).collect();
    let n: f64 = rt.iter().sum();
    let row_p = cmd.has_opt("row", 3);
    let col_p = cmd.has_opt("column", 3);
    let cell_p = cmd.has_opt("cell", 4);
    let expected = cmd.has_opt("expected", 3);
    let mut out = Vec::new();
    if n == 0.0 {
        return Ok("no observations".into());
    }
    let cw = 10;
    let rlab = if a.label.is_empty() || !labels { v1.to_string() } else { a.label.clone() };
    let clab = if b.label.is_empty() || !labels { v2.to_string() } else { b.label.clone() };
    out.push(format!("{:>11} | {:^w$} |", "", trunc(&clab, nc * (cw + 1)), w = nc * (cw + 1) - 1));
    out.push(format!(
        "{:>11} | {} | {:>10}",
        trunc(&rlab, 11),
        cc.iter().map(|c| format!("{:>w$}", trunc(&c.1, cw), w = cw)).collect::<Vec<_>>().join(" "),
        "Total"
    ));
    let sep = format!("{}+{}+{}", "-".repeat(12), "-".repeat(nc * (cw + 1) + 1), "-".repeat(11));
    out.push(sep.clone());
    let pct_lines = |vals: &[f64], tot: f64, denom: &dyn Fn(usize) -> f64, total_den: f64| -> String {
        format!(
            "{:>11} | {} | {:>10.2}",
            "",
            vals.iter().enumerate().map(|(j, v)| format!("{:>w$.2}", 100.0 * v / denom(j), w = cw)).collect::<Vec<_>>().join(" "),
            100.0 * tot / total_den
        )
    };
    for r in 0..nr {
        out.push(format!(
            "{:>11} | {} | {:>10}",
            trunc(&rc[r].1, 11),
            t[r].iter().map(|v| format!("{:>w$}", fmt_g(*v, cw), w = cw)).collect::<Vec<_>>().join(" "),
            fmt_g(rt[r], 10)
        ));
        if expected {
            let e: Vec<f64> = (0..nc).map(|c| rt[r] * ct[c] / n).collect();
            out.push(format!("{:>11} | {} | {:>10.1}", "", e.iter().map(|v| format!("{:>w$.1}", v, w = cw)).collect::<Vec<_>>().join(" "), rt[r]));
        }
        if row_p {
            out.push(pct_lines(&t[r], rt[r], &|_| rt[r], rt[r]));
        }
        if col_p {
            out.push(pct_lines(&t[r], rt[r], &|j| ct[j], n));
        }
        if cell_p {
            out.push(pct_lines(&t[r], rt[r], &|_| n, n));
        }
    }
    out.push(sep);
    out.push(format!(
        "{:>11} | {} | {:>10}",
        "Total",
        ct.iter().map(|v| format!("{:>w$}", fmt_g(*v, cw), w = cw)).collect::<Vec<_>>().join(" "),
        fmt_g(n, 10)
    ));
    if row_p {
        out.push(pct_lines(&ct, n, &|_| n, n));
    }
    if col_p {
        out.push(pct_lines(&ct, n, &|j| ct[j], n));
    }
    if cell_p {
        out.push(pct_lines(&ct, n, &|_| n, n));
    }
    let mut r = Results::new();
    r.set("N", n);
    r.set("r", nr as f64);
    r.set("c", nc as f64);
    let df = ((nr - 1) * (nc - 1)) as f64;
    let mut stats_lines = Vec::new();
    let chi2 = (0..nr)
        .flat_map(|i| (0..nc).map(move |j| (i, j)))
        .map(|(i, j)| {
            let e = rt[i] * ct[j] / n;
            if e > 0.0 { (t[i][j] - e).powi(2) / e } else { 0.0 }
        })
        .sum::<f64>();
    if cmd.has_opt("chi2", 2) {
        let p = dist::chi2tail(df, chi2);
        stats_lines.push(format!("{:>27}({}) = {:>8.4}   Pr = {:.3}", "Pearson chi2", df as i64, chi2, p));
        r.set("chi2", chi2);
        r.set("p", p);
    }
    if cmd.has_opt("lrchi2", 3) {
        let g2 = 2.0 * (0..nr)
            .flat_map(|i| (0..nc).map(move |j| (i, j)))
            .map(|(i, j)| if t[i][j] > 0.0 { t[i][j] * (t[i][j] * n / (rt[i] * ct[j])).ln() } else { 0.0 })
            .sum::<f64>();
        let p = dist::chi2tail(df, g2);
        stats_lines.push(format!("{:>27}({}) = {:>8.4}   Pr = {:.3}", "likelihood-ratio chi2", df as i64, g2, p));
        r.set("chi2_lr", g2);
        r.set("p_lr", p);
    }
    if cmd.has_opt("V", 1) || cmd.has_opt("v", 1) {
        let k = (nr.min(nc) - 1) as f64;
        let mut v = (chi2 / n / k).sqrt();
        if nr == 2 && nc == 2 {
            let d = t[0][0] * t[1][1] - t[0][1] * t[1][0];
            v = d / (rt[0] * rt[1] * ct[0] * ct[1]).sqrt();
        }
        stats_lines.push(format!("{:>30} = {:>8.4}", "Cramér's V", v));
        r.set("CramersV", v);
    }
    if cmd.has_opt("exact", 1) {
        if nr == 2 && nc == 2 {
            let (two, one) = fisher_2x2([[t[0][0], t[0][1]], [t[1][0], t[1][1]]]);
            stats_lines.push(format!("{:>30} = {:>8.3}", "Fisher's exact", two));
            stats_lines.push(format!("{:>30} = {:>8.3}", "1-sided Fisher's exact", one));
            r.set("p_exact", two);
            r.set("p1_exact", one);
        } else {
            let p = fisher_rxc(&t);
            stats_lines.push(format!("{:>30} = {:>8.3}", "Fisher's exact", p));
            r.set("p_exact", p);
        }
    }
    if cmd.has_opt("gamma", 2) || cmd.has_opt("taub", 2) {
        let (mut conc, mut disc) = (0.0, 0.0);
        for i in 0..nr {
            for j in 0..nc {
                for k in i + 1..nr {
                    for l in 0..nc {
                        if l > j {
                            conc += t[i][j] * t[k][l];
                        } else if l < j {
                            disc += t[i][j] * t[k][l];
                        }
                    }
                }
            }
        }
        if cmd.has_opt("gamma", 2) {
            let g = (conc - disc) / (conc + disc);
            stats_lines.push(format!("{:>30} = {:>8.4}", "gamma", g));
            r.set("gamma", g);
        }
        if cmd.has_opt("taub", 2) {
            let tr = n * n - rt.iter().map(|x| x * x).sum::<f64>();
            let tc = n * n - ct.iter().map(|x| x * x).sum::<f64>();
            let tb = 2.0 * (conc - disc) / (tr * tc).sqrt();
            stats_lines.push(format!("{:>30} = {:>8.4}", "Kendall's tau-b", tb));
            r.set("taub", tb);
        }
    }
    if !stats_lines.is_empty() {
        out.push(String::new());
        out.extend(stats_lines);
    }
    if let Some(m) = cmd.opt("matcell", 5) {
        st.matrices.insert(m.trim().to_string(), Matrix::from_rows(&t));
    }
    if let Some(m) = cmd.opt("matrow", 5) {
        st.matrices.insert(m.trim().to_string(), Matrix::col_vector(&rc.iter().map(|c| c.0.num()).collect::<Vec<_>>()));
    }
    if let Some(m) = cmd.opt("matcol", 5) {
        st.matrices.insert(m.trim().to_string(), Matrix::row_vector(&cc.iter().map(|c| c.0.num()).collect::<Vec<_>>()));
    }
    st.r = r;
    Ok(out.join("\n"))
}

/// Fisher's exact test for r x c tables by enumerating tables with fixed
/// margins (feasible for small tables).
fn fisher_rxc(t: &[Vec<f64>]) -> f64 {
    let nr = t.len();
    let nc = t[0].len();
    let rt: Vec<i64> = t.iter().map(|r| r.iter().sum::<f64>() as i64).collect();
    let ct: Vec<i64> = (0..nc).map(|c| t.iter().map(|r| r[c]).sum::<f64>() as i64).collect();
    let n: i64 = rt.iter().sum();
    let lnf = |x: i64| dist::lnfactorial(x as f64);
    let base: f64 = rt.iter().map(|&x| lnf(x)).sum::<f64>() + ct.iter().map(|&x| lnf(x)).sum::<f64>() - lnf(n);
    let prob = |cells: &[Vec<i64>]| -> f64 { (base - cells.iter().flatten().map(|&x| lnf(x)).sum::<f64>()).exp() };
    let obs: Vec<Vec<i64>> = t.iter().map(|r| r.iter().map(|&x| x as i64).collect()).collect();
    let p0 = prob(&obs);
    let mut total = 0.0;
    let mut cells = vec![vec![0i64; nc]; nr];
    let mut count = 0u64;
    fn rec(
        r: usize, c: usize, nr: usize, nc: usize, rows_left: &mut Vec<i64>, cols_left: &mut Vec<i64>, cells: &mut Vec<Vec<i64>>,
        f: &mut dyn FnMut(&Vec<Vec<i64>>), count: &mut u64,
    ) {
        if *count > 5_000_000 {
            return;
        }
        if r == nr - 1 {
            // last row determined by column remainders
            for j in 0..nc {
                cells[r][j] = cols_left[j];
            }
            if cols_left.iter().sum::<i64>() == rows_left[r] {
                *count += 1;
                f(cells);
            }
            return;
        }
        if c == nc - 1 {
            let v = rows_left[r];
            if v <= cols_left[c] {
                cells[r][c] = v;
                cols_left[c] -= v;
                let save = rows_left[r];
                rows_left[r] = 0;
                rec(r + 1, 0, nr, nc, rows_left, cols_left, cells, f, count);
                rows_left[r] = save;
                cols_left[c] += v;
            }
            return;
        }
        let maxv = rows_left[r].min(cols_left[c]);
        for v in 0..=maxv {
            cells[r][c] = v;
            rows_left[r] -= v;
            cols_left[c] -= v;
            rec(r, c + 1, nr, nc, rows_left, cols_left, cells, f, count);
            rows_left[r] += v;
            cols_left[c] += v;
        }
    }
    let mut rows_left = rt.clone();
    let mut cols_left = ct.clone();
    let mut f = |cells: &Vec<Vec<i64>>| {
        let p = prob(cells);
        if p <= p0 * (1.0 + 1e-7) {
            total += p;
        }
    };
    rec(0, 0, nr, nc, &mut rows_left, &mut cols_left, &mut cells, &mut f, &mut count);
    total.min(1.0)
}

/// tab1: one-way tables for each variable; tab2: all two-way combinations.
pub struct Tab1;

impl CommandHandler for Tab1 {
    fn name(&self) -> &'static str {
        "tab1"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let mut out = Vec::new();
        for v in vars {
            out.push(format!("-> tabulation of {v}"));
            out.push(String::new());
            out.push(one_way(ds, st, cmd, &v)?);
            out.push(String::new());
        }
        Ok(out.join("\n"))
    }
}

pub struct Tab2;

impl CommandHandler for Tab2 {
    fn name(&self) -> &'static str {
        "tab2"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let mut out = Vec::new();
        for i in 0..vars.len() {
            for j in i + 1..vars.len() {
                out.push(format!("-> tabulation of {} by {}", vars[i], vars[j]));
                out.push(String::new());
                out.push(two_way(ds, st, cmd, &vars[i], &vars[j])?);
                out.push(String::new());
            }
        }
        Ok(out.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fisher() {
        // classic tea-tasting table
        let (two, _) = fisher_2x2([[3.0, 1.0], [1.0, 3.0]]);
        assert!((two - 0.4857142857).abs() < 1e-8);
        let p = fisher_rxc(&[vec![3.0, 1.0], vec![1.0, 3.0]]);
        assert!((p - 0.4857142857).abs() < 1e-8);
    }
}
