use ferrostat_core::DataSet;
use ferrostat_parser::{Command, Expr, InRange};
use crate::commands::CommandHandler;
use crate::eval::{eval_expr, EvalResult};

pub struct Summarize;

impl CommandHandler for Summarize {
    fn name(&self) -> &'static str { "summarize" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        let varlist: Vec<String> = if cmd.args.is_empty() {
            ds.varlist()
        } else {
            cmd.args.clone()
        };

        let mut selected_rows: Vec<usize> = (0..ds.n_obs).collect();
        if let Some(ref range) = cmd.in_range {
            selected_rows = apply_in(ds, range)?;
        }
        if let Some(ref expr) = cmd.if_expr {
            let mask_rows: std::collections::HashSet<usize> = apply_if(ds, expr)?.into_iter().collect();
            selected_rows.retain(|r| mask_rows.contains(r));
        }

        let mut out = Vec::new();
        out.push(format!("    Variable |       Obs        Mean    Std. Dev.       Min        Max"));
        out.push("-".repeat(72));

        for name in &varlist {
            let var = match ds.get_var(name) {
                Some(v) => v,
                None => continue,
            };
            if var.var_type.is_string() {
                continue;
            }

            let vals: Vec<f64> = selected_rows.iter()
                .filter_map(|&i| var.data[i].as_f64())
                .filter(|v| !v.is_nan())
                .collect();

            let obs = vals.len();
            if obs == 0 {
                out.push(format!("{:>11} |         0", name));
                continue;
            }

            let mean = vals.iter().sum::<f64>() / obs as f64;
            let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (obs as f64 - 1.0).max(1.0);
            let stddev = variance.sqrt();
            let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            out.push(format!(
                "{:>11} | {:>9} {:>11.6} {:>11.6} {:>10.6} {:>10.6}",
                name, obs, mean, stddev, min, max
            ));
        }

        Ok(out.join("\n"))
    }
}

/// Apply an `if` expression to select row indices.
pub fn apply_if(ds: &DataSet, expr: &Expr) -> Result<Vec<usize>, String> {
    let mut result = Vec::new();
    for i in 0..ds.n_obs {
        match eval_expr(expr, ds, i) {
            Ok(val) => {
                if val.as_bool() && !val.is_missing() {
                    result.push(i);
                }
            }
            Err(_) => { /* expression error on this row — skip */ }
        }
    }
    Ok(result)
}

/// Apply an `in` range to select row indices (1-based).
pub fn apply_in(ds: &DataSet, range: &InRange) -> Result<Vec<usize>, String> {
    match range {
        InRange::Single(n) => {
            let idx = n.saturating_sub(1);
            if idx < ds.n_obs { Ok(vec![idx]) } else { Ok(vec![]) }
        }
        InRange::Range(start, end) => {
            let s = start.saturating_sub(1);
            let e = (*end).min(ds.n_obs);
            Ok((s..e).collect())
        }
        InRange::First(n) => {
            let e = (*n).min(ds.n_obs);
            Ok((0..e).collect())
        }
        InRange::Last(n) => {
            let start = ds.n_obs.saturating_sub(*n);
            Ok((start..ds.n_obs).collect())
        }
    }
}
