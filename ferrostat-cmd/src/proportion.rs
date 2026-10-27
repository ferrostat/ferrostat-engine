//! `proportion varlist [if] [in] [fw] [, over(varlist) citype(logit|wald)
//! level(#) missing nolabel]`: the proportion of observations in each level,
//! with standard errors and (by default logit-transformed) confidence
//! intervals; posts e(b), e(V) and e(N).

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::dist;
use crate::eval::rc_err;
use crate::format::fmt_g;
use crate::matrix::Matrix;
use crate::store::Store;
use crate::summarize::abbrev;
use crate::{util, varlist};

pub struct Proportion;

impl CommandHandler for Proportion {
    fn name(&self) -> &'static str {
        "proportion"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.is_empty() {
            return Err(rc_err(100, "varlist required"));
        }
        let missing = cmd.has_opt("missing", 4);
        let labels = !cmd.has_opt("nolabel", 5);
        let logit = match cmd.opt("citype", 4).map(|c| c.trim().to_lowercase()) {
            None => true,
            Some(c) if c == "logit" => true,
            Some(c) if c == "wald" || c == "normal" => false,
            Some(c) => return Err(rc_err(198, format!("citype({c}) not allowed"))),
        };
        let mut mask = util::touse(ds, st, cmd)?;
        let w = util::weights(ds, st, cmd, &mut mask, "f")?;
        if !missing {
            util::markout(ds, &mut mask, &vars);
        }
        let over: Vec<String> = match cmd.opt("over", 2) {
            Some(o) => varlist::expand(ds, o.split(',').next().unwrap_or(""))?,
            None => Vec::new(),
        };
        util::markout(ds, &mut mask, &over);
        let level = util::level(cmd, st)?;
        let rows = util::rows(&mask);
        if rows.is_empty() {
            return Err(rc_err(2000, "no observations"));
        }
        let wt = |i: usize| w.as_ref().map_or(1.0, |w| w.w[i]);
        let n: f64 = rows.iter().map(|&i| wt(i)).sum();
        let (gids, greps) = if over.is_empty() { (vec![0; ds.n_obs], vec![rows[0]]) } else { util::group_ids(ds, &over, &mask) };
        let ng = greps.len();
        let gname = |g: usize| over.iter().map(|v| format!("{}.{v}", fmt_g(ds.get_var(v).unwrap().num(greps[g]), 9))).collect::<Vec<_>>().join("#");
        let glabel = |g: usize| over.iter().map(|v| util::cell_text(ds, ds.get_var(v).unwrap(), greps[g], labels)).collect::<Vec<_>>().join(" ");

        // one estimate per (variable, level, over group)
        struct Est {
            /// variable and over group: levels within one share a covariance
            key: (usize, usize),
            n: f64,
            name: String,
            label: String,
            p: f64,
            se: f64,
        }
        let mut blocks: Vec<(String, Vec<Est>)> = Vec::new();
        for (vi, v) in vars.iter().enumerate() {
            let var = ds.get_var(v).unwrap();
            if var.var_type.is_string() {
                return Err(rc_err(109, format!("{v}: string variables not allowed")));
            }
            let mut levels: Vec<f64> = rows.iter().map(|&i| var.num(i)).collect();
            levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
            levels.dedup();
            let mut ests = Vec::new();
            for &lev in &levels {
                for g in 0..ng {
                    let in_g: Vec<usize> = rows.iter().copied().filter(|&i| gids[i] == g).collect();
                    let ng_w: f64 = in_g.iter().map(|&i| wt(i)).sum();
                    let hits: f64 = in_g.iter().filter(|&&i| var.num(i) == lev).map(|&i| wt(i)).sum();
                    let p = hits / ng_w;
                    let se = (p * (1.0 - p) / (ng_w - 1.0)).sqrt();
                    let levtext = if ferrostat_core::missing::is_missing(lev) { ".".to_string() } else { fmt_g(lev, 9) };
                    let labtext = util::cell_text(ds, var, rows.iter().copied().find(|&i| var.num(i) == lev).unwrap_or(rows[0]), labels);
                    let name = if over.is_empty() { format!("{levtext}.{v}") } else { format!("{levtext}.{v}@{}", gname(g)) };
                    let label = if over.is_empty() { labtext } else { format!("{labtext} {}", glabel(g)) };
                    ests.push(Est { key: (vi, g), n: ng_w, name, label, p, se });
                }
            }
            let head = if over.is_empty() { v.clone() } else { format!("{v}@{}", over.join("#")) };
            blocks.push((head, ests));
        }

        let df = n - 1.0;
        let tcrit = dist::invttail(df, (1.0 - level / 100.0) / 2.0);
        let ci = |p: f64, se: f64| -> (f64, f64) {
            if logit && p > 0.0 && p < 1.0 {
                let l = (p / (1.0 - p)).ln();
                let s = se / (p * (1.0 - p));
                let back = |x: f64| 1.0 / (1.0 + (-x).exp());
                (back(l - tcrit * s), back(l + tcrit * s))
            } else if logit {
                (p, p)
            } else {
                (p - tcrit * se, p + tcrit * se)
            }
        };
        let width = blocks.iter().flat_map(|(h, e)| std::iter::once(h.len()).chain(e.iter().map(|x| x.label.len()))).max().unwrap_or(12).clamp(12, 32);
        let dash = |k: usize| "-".repeat(k);
        let mut out = vec![format!("{:<w2$}Number of obs = {:>6}", "Proportion estimation", fmt_g(n, 9), w2 = width + 30), String::new()];
        out.push(dash(width + 50));
        out.push(format!("{:w$} |{:>40}", "", if logit { "Logit" } else { "" }, w = width).trim_end().to_string());
        out.push(format!("{:w$} | Proportion   Std. err.     [{}% conf. interval]", "", fmt_g(level, 4), w = width));
        out.push(format!("{}+{}", dash(width + 1), dash(48)));
        let mut names = Vec::new();
        let mut b = Vec::new();
        let mut keys = Vec::new();
        for (head, ests) in &blocks {
            out.push(format!("{:>w$} |", abbrev(head, width), w = width));
            for e in ests {
                let (lo, hi) = ci(e.p, e.se);
                out.push(format!("{:>w$} | {:>10} {:>10}  {:>12} {:>11}", abbrev(&e.label, width), fmt_g(e.p, 8), fmt_g(e.se, 8), fmt_g(lo, 8), fmt_g(hi, 8), w = width));
                names.push(e.name.clone());
                b.push(e.p);
                keys.push((e.key, e.n));
            }
        }
        out.push(dash(width + 50));

        let k = b.len();
        let mut v = Matrix::zeros(k, k);
        // multinomial covariance: p_i (δ_ij - p_j) / (n - 1) within a variable and group
        for i in 0..k {
            for j in 0..k {
                if keys[i].0 == keys[j].0 {
                    let d = if i == j { 1.0 } else { 0.0 };
                    v[(i, j)] = b[i] * (d - b[j]) / (keys[i].1 - 1.0);
                }
            }
        }
        let bm = Matrix::row_vector(&b).with_names(vec!["y1".into()], names.clone());
        let vm = v.with_names(names.clone(), names);
        let e = st.begin_e();
        e.set("N", n);
        e.set("df_r", df);
        e.set("level", level);
        e.set_str("cmd", "proportion");
        e.set_str("title", "Proportion estimation");
        e.set_str("citype", if logit { "logit" } else { "wald" });
        e.set_str("varlist", vars.join(" "));
        if !over.is_empty() {
            e.set_str("over", over.join(" "));
        }
        e.set_mat("b", bm);
        e.set_mat("V", vm);
        Ok(out.join("\n"))
    }
}
