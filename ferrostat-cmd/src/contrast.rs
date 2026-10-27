//! contrast: contrasts of marginal linear predictions with joint Wald tests.
//!
//! Margins are computed "as balanced": factor variables not fixed by a cell
//! are averaged evenly over their levels and continuous covariates are held
//! at their estimation-sample means. Every margin is linear in the
//! coefficients, so each contrast is a row `L` with estimate `L b` and
//! variance `L V L'`.

use std::collections::BTreeMap;

use ferrostat_core::missing::is_missing;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::{Results, Store};
use crate::summarize::abbrev;
use crate::util;

pub struct Contrast;

/// Contrast operator applied to one factor of a term.
#[derive(Clone, Debug, PartialEq)]
enum Op {
    /// No operator: only the joint test (reference contrasts internally).
    Test,
    /// r. / rb#. : each level against the reference level.
    Ref(Option<f64>),
    /// a. : each level against the next one.
    Adjacent,
    /// ar. : each level against the previous one.
    RevAdjacent,
    /// g. : each level against the grand mean of all levels.
    Grand,
}

/// One contrast of a factor: weights over its levels plus name/label pieces.
struct Piece {
    weights: Vec<f64>,
    name: String,
    label: String,
}

struct Term {
    parts: Vec<(Op, String)>,
    over: Vec<String>,
}

fn parse_term(t: &str) -> Result<Term, String> {
    let (left, right) = match t.split_once('@') {
        Some((l, r)) => (l, Some(r)),
        None => (t, None),
    };
    let mut parts = Vec::new();
    for p in left.split('#') {
        let (op, var) = match p.split_once('.') {
            Some((op, var)) => {
                let o = op.to_lowercase();
                let op = match o.as_str() {
                    "i" | "" => Op::Test,
                    "r" => Op::Ref(None),
                    "a" => Op::Adjacent,
                    "ar" => Op::RevAdjacent,
                    "g" | "gw" => Op::Grand,
                    _ if o.starts_with("rb") => Op::Ref(Some(o[2..].parse().map_err(|_| rc_err(198, format!("{p}: invalid contrast operator")))?)),
                    _ => return Err(rc_err(198, format!("{op}.: unknown contrast operator"))),
                };
                (op, var.to_string())
            }
            None => (Op::Test, p.to_string()),
        };
        parts.push((op, var));
    }
    let over = right.map_or(Vec::new(), |r| r.split('#').map(|v| v.split_once('.').map_or(v, |(_, x)| x).to_string()).collect());
    Ok(Term { parts, over })
}

/// Parse one part of a coefficient name: Some((level, var)) for a factor.
fn factor_part(p: &str) -> Option<(f64, &str)> {
    let p = p.strip_prefix("o.").unwrap_or(p);
    let (l, v) = p.split_once('.')?;
    Some((l.trim_end_matches(['b', 'o', 'n']).parse().ok()?, v))
}

fn bare(c: &str) -> &str {
    c.rsplit(':').next().unwrap_or(c)
}

/// Levels of factor `var` among the coefficient names, and its base level.
fn factor_levels(b: &Matrix, var: &str) -> (Vec<f64>, Option<f64>) {
    let mut lv = Vec::new();
    let mut base = None;
    for c in &b.colnames {
        for p in bare(c).split('#') {
            if let Some((l, v)) = factor_part(p) {
                if v == var {
                    lv.push(l);
                    let lt = p.split_once('.').unwrap().0;
                    if lt.ends_with('b') && !bare(c).contains('#') {
                        base = Some(l);
                    }
                }
            }
        }
    }
    lv.sort_by(f64::total_cmp);
    lv.dedup();
    (lv, base)
}

fn level_label(ds: &DataSet, var: &str, l: f64) -> String {
    ds.get_var(var)
        .filter(|v| !v.value_label_name.is_empty())
        .and_then(|v| ds.value_labels.get(&v.value_label_name).and_then(|vl| vl.lookup(l)).map(String::from))
        .unwrap_or_else(|| fmt_g(l, 12))
}

fn pieces(ds: &DataSet, op: &Op, var: &str, levels: &[f64], base: Option<f64>) -> Result<Vec<Piece>, String> {
    let n = levels.len();
    let lab = |l: f64| level_label(ds, var, l);
    let num = |l: f64| fmt_g(l, 12);
    let unit = |i: usize| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect::<Vec<f64>>();
    let diff = |i: usize, j: usize| unit(i).iter().zip(unit(j)).map(|(a, b)| a - b).collect::<Vec<f64>>();
    let mut out = Vec::new();
    match op {
        Op::Test | Op::Ref(_) => {
            let r = match op {
                Op::Ref(Some(r)) => *r,
                _ => base.unwrap_or(levels[0]),
            };
            let ri = levels.iter().position(|&l| l == r).ok_or_else(|| rc_err(198, format!("level {} of {var} not found", num(r))))?;
            for i in 0..n {
                if i != ri {
                    out.push(Piece { weights: diff(i, ri), name: format!("r{}vs{}", num(levels[i]), num(r)), label: format!("({} vs {})", lab(levels[i]), lab(r)) });
                }
            }
        }
        Op::Adjacent => {
            for i in 0..n.saturating_sub(1) {
                out.push(Piece { weights: diff(i, i + 1), name: format!("a{}vs{}", num(levels[i]), num(levels[i + 1])), label: format!("({} vs {})", lab(levels[i]), lab(levels[i + 1])) });
            }
        }
        Op::RevAdjacent => {
            for i in 1..n {
                out.push(Piece { weights: diff(i, i - 1), name: format!("ar{}vs{}", num(levels[i]), num(levels[i - 1])), label: format!("({} vs {})", lab(levels[i]), lab(levels[i - 1])) });
            }
        }
        Op::Grand => {
            for i in 0..n {
                let w = (0..n).map(|j| if i == j { 1.0 } else { 0.0 } - 1.0 / n as f64).collect();
                out.push(Piece { weights: w, name: format!("g{}", num(levels[i])), label: format!("({} vs mean)", lab(levels[i])) });
            }
        }
    }
    Ok(out)
}

/// Cartesian product of index ranges.
fn product(sizes: &[usize]) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new()];
    for &s in sizes {
        out = out.into_iter().flat_map(|p| (0..s).map(move |i| [p.clone(), vec![i]].concat())).collect();
    }
    out
}

struct Model<'a> {
    b: &'a Matrix,
    levels: BTreeMap<String, Vec<f64>>,
    means: BTreeMap<String, f64>,
    eq: Option<String>,
}

impl Model<'_> {
    /// Gradient of the balanced margin at fixed factor levels.
    fn cell_row(&self, fixed: &BTreeMap<String, f64>) -> Vec<f64> {
        self.b
            .colnames
            .iter()
            .map(|c| {
                if let (Some(eq), Some((e, _))) = (&self.eq, c.split_once(':')) {
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
                            None => prod /= self.levels.get(v).map_or(1, |x| x.len().max(1)) as f64,
                        },
                        None => {
                            let v = p.strip_prefix("o.").unwrap_or(p);
                            let v = v.strip_prefix("c.").unwrap_or(v);
                            prod *= self.means.get(v).copied().unwrap_or(0.0);
                        }
                    }
                }
                prod
            })
            .collect()
    }
}

/// Wald test of `L b = 0`: (statistic, df).
fn wald(lb: &[f64], lvl: &Matrix) -> (f64, f64) {
    let q = lb.len();
    let (inv, dropped) = lvl.sym_inverse(1e-12);
    let w: f64 = (0..q).map(|a| (0..q).map(|c| lb[a] * inv[(a, c)] * lb[c]).sum::<f64>()).sum();
    (w, (q - dropped.len()) as f64)
}

impl CommandHandler for Contrast {
    fn name(&self) -> &'static str {
        "contrast"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let b = st.e.matrices.get("b").ok_or_else(|| rc_err(301, "last estimates not found"))?.clone();
        let v = st.e.matrices.get("V").ok_or_else(|| rc_err(301, "last estimates not found"))?.clone();
        let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
        let level = util::level(cmd, st)?;
        let terms: Vec<Term> = cmd.main.split_whitespace().map(parse_term).collect::<Result<_, _>>()?;
        if terms.is_empty() {
            return Err(rc_err(198, "contrast: term required"));
        }
        // factor levels and covariate means over the estimation sample
        let sample: Vec<usize> = st.e.sample.as_ref().map_or_else(Vec::new, |m| util::rows(m));
        let mut levels = BTreeMap::new();
        let mut means = BTreeMap::new();
        for c in &b.colnames {
            for p in bare(c).split('#') {
                match factor_part(p) {
                    Some((_, var)) => {
                        levels.entry(var.to_string()).or_insert_with(|| factor_levels(&b, var).0);
                    }
                    None if p != "_cons" && !p.starts_with('/') => {
                        let var = p.strip_prefix("o.").unwrap_or(p);
                        let var = var.strip_prefix("c.").unwrap_or(var).to_string();
                        if !means.contains_key(&var) {
                            let m = util::col(ds, &var)
                                .map(|x| {
                                    let vals: Vec<f64> = sample.iter().map(|&i| x[i]).filter(|x| !is_missing(*x)).collect();
                                    util::mean(&vals)
                                })
                                .unwrap_or(0.0);
                            means.insert(var, m);
                        }
                    }
                    None => {}
                }
            }
        }
        let eq = b.colnames.first().and_then(|c| c.split_once(':')).map(|(e, _)| e.to_string());
        let model = Model { b: &b, levels, means, eq };
        let k = b.cols;
        let df_r = st.e.get("df_r").filter(|_| matches!(ecmd.as_str(), "regress" | "anova" | "areg" | "xtreg" | "mean") && st.e.macros.get("model").map_or(true, |m| m != "re"));
        let stat = if df_r.is_some() { "F" } else { "chi2" };

        // contrasts: (term index, over-cell label, name, label, L row)
        struct Row {
            term: usize,
            cell: usize,
            name: String,
            label: String,
            l: Vec<f64>,
        }
        let mut rows: Vec<Row> = Vec::new();
        // per term: (header, show contrasts, over-cell labels)
        let mut heads: Vec<(String, bool, Vec<String>)> = Vec::new();
        for (ti, t) in terms.iter().enumerate() {
            let mut part_pieces = Vec::new();
            let mut part_levels = Vec::new();
            for (op, var) in &t.parts {
                let (lv, base) = factor_levels(&b, var);
                if lv.is_empty() {
                    return Err(rc_err(111, format!("factor variable {var} not found in the model")));
                }
                part_pieces.push(pieces(ds, op, var, &lv, base)?);
                part_levels.push(lv);
            }
            let mut over_levels = Vec::new();
            for o in &t.over {
                let (lv, _) = factor_levels(&b, o);
                if lv.is_empty() {
                    return Err(rc_err(111, format!("factor variable {o} not found in the model")));
                }
                over_levels.push(lv);
            }
            let cells = product(&over_levels.iter().map(Vec::len).collect::<Vec<_>>());
            let mut cell_labels = Vec::new();
            for (ci, cell) in cells.iter().enumerate() {
                let mut fixed = BTreeMap::new();
                let mut cname = Vec::new();
                let mut clab = Vec::new();
                for (oi, &li) in cell.iter().enumerate() {
                    let l = over_levels[oi][li];
                    fixed.insert(t.over[oi].clone(), l);
                    cname.push(format!("{}.{}", fmt_g(l, 12), t.over[oi]));
                    clab.push(level_label(ds, &t.over[oi], l));
                }
                cell_labels.push(clab.join("#"));
                for combo in product(&part_pieces.iter().map(Vec::len).collect::<Vec<_>>()) {
                    let ps: Vec<&Piece> = combo.iter().enumerate().map(|(pi, &i)| &part_pieces[pi][i]).collect();
                    let mut l = vec![0.0; k];
                    for lcell in product(&part_levels.iter().map(Vec::len).collect::<Vec<_>>()) {
                        let w: f64 = lcell.iter().enumerate().map(|(pi, &li)| ps[pi].weights[li]).product();
                        if w == 0.0 {
                            continue;
                        }
                        let mut f = fixed.clone();
                        for (pi, &li) in lcell.iter().enumerate() {
                            f.insert(t.parts[pi].1.clone(), part_levels[pi][li]);
                        }
                        for (x, g) in l.iter_mut().zip(model.cell_row(&f)) {
                            *x += w * g;
                        }
                    }
                    let mut name = ps.iter().zip(&t.parts).map(|(p, (_, var))| format!("{}.{var}", p.name)).collect::<Vec<_>>().join("#");
                    if !cname.is_empty() {
                        name = format!("{name}@{}", cname.join("#"));
                    }
                    let mut label = ps.iter().map(|p| p.label.clone()).collect::<Vec<_>>().join(" ");
                    if !clab.is_empty() {
                        label = format!("{label} {}", clab.join("#"));
                    }
                    rows.push(Row { term: ti, cell: ci, name, label, l });
                }
            }
            let mut head = t.parts.iter().map(|(_, v)| v.clone()).collect::<Vec<_>>().join("#");
            if !t.over.is_empty() {
                head = format!("{head}@{}", t.over.join("#"));
            }
            heads.push((head, t.parts.iter().any(|(op, _)| *op != Op::Test), if t.over.is_empty() { Vec::new() } else { cell_labels }));
        }

        let m = rows.len();
        let mut lmat = Matrix::zeros(m, k);
        for (i, r) in rows.iter().enumerate() {
            for j in 0..k {
                lmat[(i, j)] = r.l[j];
            }
        }
        let est: Vec<f64> = rows.iter().map(|r| r.l.iter().zip(&b.data).map(|(a, x)| a * x).sum()).collect();
        let vm = lmat.mul(&v)?.mul(&lmat.transpose())?;
        let sub = |idx: &[usize]| -> (f64, f64) {
            let mut s = Matrix::zeros(idx.len(), idx.len());
            for (a, &i) in idx.iter().enumerate() {
                for (c, &j) in idx.iter().enumerate() {
                    s[(a, c)] = vm[(i, j)];
                }
            }
            let lb: Vec<f64> = idx.iter().map(|&i| est[i]).collect();
            wald(&lb, &s)
        };
        let finish = |w: f64, q: f64| -> (f64, f64) {
            match df_r {
                Some(d) => (w / q, dist::ftail(q, d, w / q)),
                None => (w, dist::chi2tail(q, w)),
            }
        };

        // tests table
        let mut test_lines = Vec::new();
        let (mut t_stat, mut t_df, mut t_p) = (Vec::new(), Vec::new(), Vec::new());
        let lw = heads.iter().map(|h| h.0.chars().count()).chain(rows.iter().map(|r| r.label.chars().count())).max().unwrap_or(12).clamp(12, 32);
        let ab = |x: &str| abbrev(x, lw);
        for (ti, (head, _, cells)) in heads.iter().enumerate() {
            if cells.is_empty() {
                let idx: Vec<usize> = (0..m).filter(|&i| rows[i].term == ti).collect();
                let (w, q) = sub(&idx);
                let (s, p) = finish(w, q);
                test_lines.push(format!("{:>lw$} |  {:>9}  {:>10}  {:>9}", ab(head), q, format!("{s:.2}"), format!("{p:.4}")));
                t_stat.push(s);
                t_df.push(q);
                t_p.push(p);
            } else {
                test_lines.push(format!("{:<lw$} |", ab(head)));
                for (ci, cl) in cells.iter().enumerate() {
                    let idx: Vec<usize> = (0..m).filter(|&i| rows[i].term == ti && rows[i].cell == ci).collect();
                    let (w, q) = sub(&idx);
                    let (s, p) = finish(w, q);
                    test_lines.push(format!("{:>lw$} |  {:>9}  {:>10}  {:>9}", ab(&format!("{cl} ")), q, format!("{s:.2}"), format!("{p:.4}")));
                    t_stat.push(s);
                    t_df.push(q);
                    t_p.push(p);
                }
                let idx: Vec<usize> = (0..m).filter(|&i| rows[i].term == ti).collect();
                let (w, q) = sub(&idx);
                let (s, p) = finish(w, q);
                test_lines.push(format!("{:>lw$} |  {:>9}  {:>10}  {:>9}", "Joint", q, format!("{s:.2}"), format!("{p:.4}")));
                t_stat.push(s);
                t_df.push(q);
                t_p.push(p);
            }
        }
        let mut out = vec!["Contrasts of marginal linear predictions".to_string(), String::new(), "Margins: asbalanced".to_string(), String::new()];
        let tw = lw + 37;
        out.push("-".repeat(tw));
        out.push(format!("{:>lw$} |         df  {:>10}  {:>9}", "", stat, if df_r.is_some() { "P>F" } else { "P>chi2" }));
        out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(35)));
        out.extend(test_lines);
        if let Some(d) = df_r {
            out.push(format!("{:>lw$} |", ""));
            out.push(format!("{:>lw$} |  {:>9}", "Denominator", d));
        }
        out.push("-".repeat(tw));

        // contrasts table
        let effects = cmd.has_opt("effects", 3);
        let crit = match df_r {
            Some(d) => dist::invttail(d, (1.0 - level / 100.0) / 2.0),
            None => dist::invnormal(1.0 - (1.0 - level / 100.0) / 2.0),
        };
        let shown: Vec<usize> = (0..m).filter(|&i| heads[rows[i].term].1).collect();
        if !shown.is_empty() {
            let zt = if df_r.is_some() { "t" } else { "z" };
            let width = if effects { 66 } else { 48 };
            out.push(String::new());
            out.push("-".repeat(lw + 2 + width));
            if effects {
                out.push(format!("{:>lw$} |   Contrast   Std. err.  {:>6}    P>|{zt}|     [{}% conf. interval]", "", zt, fmt_g(level, 4)));
            } else {
                out.push(format!("{:>lw$} |   Contrast   Std. err.     [{}% conf. interval]", "", fmt_g(level, 4)));
            }
            out.push(format!("{}+{}", "-".repeat(lw + 1), "-".repeat(width)));
            let mut last = usize::MAX;
            for &i in &shown {
                let r = &rows[i];
                if r.term != last {
                    out.push(format!("{:<lw$} |", ab(&heads[r.term].0)));
                    last = r.term;
                }
                let se = vm[(i, i)].max(0.0).sqrt();
                let (lo, hi) = (est[i] - crit * se, est[i] + crit * se);
                if effects {
                    let t = est[i] / se;
                    let p = match df_r {
                        Some(d) => 2.0 * dist::ttail(d, t.abs()),
                        None => 2.0 * (1.0 - dist::normal(t.abs())),
                    };
                    out.push(format!("{:>lw$} |  {:>9}  {:>9}  {:>7.2}   {:.3}     {:>9}   {:>9}", ab(&r.label), fmt_g(est[i], 8), fmt_g(se, 8), t, p, fmt_g(lo, 8), fmt_g(hi, 8)));
                } else {
                    out.push(format!("{:>lw$} |  {:>9}  {:>9}     {:>9}   {:>9}", ab(&r.label), fmt_g(est[i], 8), fmt_g(se, 8), fmt_g(lo, 8), fmt_g(hi, 8)));
                }
            }
            out.push("-".repeat(lw + 2 + width));
        }

        let names: Vec<String> = rows.iter().map(|r| r.name.clone()).collect();
        let no = vec![false; m];
        let table = crate::est::table_matrix(&crate::est::TableSpec { depvar: "", names: &names, b: &est, v: &vm, omitted: &no, is_base: &no, df: df_r, level, eform: None, coef_title: "Contrast" });
        let bm = Matrix::row_vector(&est).with_names(vec!["y1".into()], names.clone());
        let vmn = vm.clone().with_names(names.clone(), names.clone());
        let lm = lmat.with_names(names.clone(), b.colnames.clone());
        let mut res = Results::new();
        res.set_mat("b", bm.clone());
        res.set_mat("V", vmn.clone());
        res.set_mat("L", lm.clone());
        res.set_mat("table", table);
        res.set_mat(stat, Matrix::row_vector(&t_stat));
        res.set_mat("df", Matrix::row_vector(&t_df));
        res.set_mat("p", Matrix::row_vector(&t_p));
        res.set("level", level);
        res.set("k_terms", terms.len() as f64);
        if let Some(d) = df_r {
            res.set("df_r", d);
        }
        res.set_str("cmd", "contrast");
        if cmd.has_opt("post", 4) {
            let n = st.e.get("N");
            let sample = st.e.sample.clone();
            let depvar = st.e.macros.get("depvar").cloned().unwrap_or_default();
            let e = st.begin_e();
            if let Some(n) = n {
                e.set("N", n);
            }
            if let Some(d) = df_r {
                e.set("df_r", d);
            }
            e.set("k_terms", terms.len() as f64);
            e.set_str("cmd", "contrast");
            e.set_str("est_cmd", ecmd);
            e.set_str("depvar", depvar);
            e.set_str("title", "Contrasts of marginal linear predictions");
            e.set_str("properties", "b V");
            e.set_mat("b", bm);
            e.set_mat("V", vmn);
            e.set_mat("L", lm);
            e.sample = sample;
        }
        st.r = res;
        Ok(out.join("\n"))
    }
}
