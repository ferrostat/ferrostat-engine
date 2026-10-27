//! `unique`: number of distinct values (or value combinations) of a varlist,
//! optionally within groups of a `by()` variable.

use ferrostat_core::missing::SYSMISS;
use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::store::Store;
use crate::{util, varlist};

pub struct Unique;

impl CommandHandler for Unique {
    fn name(&self) -> &'static str {
        "unique"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.is_empty() {
            return Err(rc_err(100, "varlist required"));
        }
        let mut mask = util::touse(ds, st, cmd)?;
        util::markout(ds, &mut mask, &vars);
        let nrec = mask.iter().filter(|m| **m).count();
        let (_, reps) = util::group_ids(ds, &vars, &mask);
        let nuniq = reps.len();
        let mut out = vec![
            format!("Number of unique values of {} is  {}", vars.join(" "), nuniq),
            format!("Number of records is  {}", nrec),
        ];
        let r = st.begin_r();
        r.set("N", nrec as f64);
        r.set("unique", nuniq as f64);
        r.set("sum", nuniq as f64);
        r.set("max", nuniq as f64);
        let Some(by) = cmd.opt("by", 2) else {
            if cmd.has_opt("generate", 3) {
                return Err(rc_err(198, "generate() requires by()"));
            }
            return Ok(out.join("\n"));
        };
        // within groups of by(): count distinct combinations per group
        let byv = varlist::expand(ds, by)?;
        util::markout(ds, &mut mask, &byv);
        let (gid, greps) = util::group_ids(ds, &byv, &mask);
        let mut both = byv.clone();
        both.extend(vars.iter().cloned());
        let (_, creps) = util::group_ids(ds, &both, &mask);
        let mut counts = vec![0usize; greps.len()];
        for &i in &creps {
            counts[gid[i]] += 1;
        }
        let newvar = cmd.opt("generate", 3).map(|g| g.trim().to_string()).unwrap_or_else(|| "_Unique".to_string());
        if ds.get_var(&newvar).is_some() {
            return Err(rc_err(110, format!("variable {newvar} already defined")));
        }
        // the count is stored on the first observation of each group
        let mut vals = vec![SYSMISS; ds.n_obs];
        for (g, &i) in greps.iter().enumerate() {
            let first = (0..ds.n_obs).find(|&j| gid[j] == g).unwrap_or(i);
            vals[first] = counts[g] as f64;
        }
        let mut var = util::new_double(&newvar, vals);
        var.format = "%9.0g".into();
        ds.add_variable(var)?;
        let sum: usize = counts.iter().sum();
        let max = counts.iter().copied().max().unwrap_or(0);
        let r = st.begin_r();
        r.set("N", nrec as f64);
        r.set("unique", nuniq as f64);
        r.set("sum", sum as f64);
        r.set("max", max as f64);
        out.push(format!("variable {newvar} contains number of unique values of {} by {}", vars.join(" "), byv.join(" ")));
        let w = byv.iter().map(|b| b.len().max(8)).collect::<Vec<_>>();
        let mut head = String::new();
        for (b, wd) in byv.iter().zip(&w) {
            head.push_str(&format!("{:>wd$}  ", b, wd = wd));
        }
        head.push_str(&format!("{:>8}", newvar));
        out.push(String::new());
        out.push(head);
        for (g, &i) in greps.iter().enumerate() {
            let mut line = String::new();
            for (b, wd) in byv.iter().zip(&w) {
                let v = ds.get_var(b).unwrap();
                line.push_str(&format!("{:>wd$}  ", util::cell_text(ds, v, i, true), wd = wd));
            }
            line.push_str(&format!("{:>8}", counts[g]));
            out.push(line);
        }
        if cmd.has_opt("detail", 1) {
            let mut c: Vec<f64> = counts.iter().map(|&x| x as f64).collect();
            c.sort_by(f64::total_cmp);
            out.push(String::new());
            out.push(format!("Unique values of {} per group of {}: n = {}, mean = {}, min = {}, median = {}, max = {}", vars.join(" "), byv.join(" "), c.len(), crate::format::fmt_g(util::mean(&c), 6), c.first().copied().unwrap_or(0.0), crate::format::fmt_g(util::pctile_sorted(&c, 50.0), 6), max));
        }
        Ok(out.join("\n"))
    }
}
