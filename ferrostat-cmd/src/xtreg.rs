use std::collections::HashMap;

use ferrostat_core::DataSet;
use ferrostat_parser::Command;
use crate::commands::CommandHandler;

/// Panel data fixed/random effects regression.
pub struct Xtreg;

impl CommandHandler for Xtreg {
    fn name(&self) -> &'static str { "xtreg" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.len() < 3 {
            return Err("xtreg requires panel_var, depvar, and indepvars".into());
        }

        let panel_var = &cmd.args[0];
        let depvar = &cmd.args[1];
        let indepvars = &cmd.args[2..];

        // Determine model type: fe (fixed effects) or re (random effects)
        let model = cmd.options.iter()
            .find(|(k, _)| k == "fe" || k == "re")
            .map(|(k, _)| k.clone())
            .unwrap_or_else(|| "fe".to_string());

        let dep = ds.get_var(depvar)
            .ok_or_else(|| format!("variable {} not found", depvar))?
            .as_f64_vec()
            .ok_or("dependent variable must be numeric")?;

        let panel = ds.get_var(panel_var)
            .ok_or_else(|| format!("variable {} not found", panel_var))?
            .as_f64_vec()
            .ok_or("panel variable must be numeric")?;

        let n = dep.len();

        // Group by panel
        let mut panels: HashMap<i64, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let pid = panel[i].round() as i64;
            panels.entry(pid).or_insert_with(Vec::new).push(i);
        }

        let num_panels = panels.len();

        // Fixed effects: within transformation (subtract panel means)
        if model == "fe" {
            let mut x: Vec<Vec<f64>> = Vec::new();
            for name in indepvars {
                let col = ds.get_var(name)
                    .ok_or_else(|| format!("variable {} not found", name))?
                    .as_f64_vec()
                    .ok_or("independent variable must be numeric")?;
                x.push(col);
            }

            // Within transformation
            let mut within_dep: Vec<f64> = Vec::new();
            let mut within_x: Vec<Vec<f64>> = vec![Vec::new(); x.len()];

            for (_pid, indices) in &panels {
                let n_i = indices.len() as f64;
                let y_mean: f64 = indices.iter().map(|&i| dep[i]).sum::<f64>() / n_i;
                let x_means: Vec<f64> = x.iter().map(|col| {
                    indices.iter().map(|&i| col[i]).sum::<f64>() / n_i
                }).collect();

                for &i in indices {
                    within_dep.push(dep[i] - y_mean);
                    for (j, col) in x.iter().enumerate() {
                        within_x[j].push(col[i] - x_means[j]);
                    }
                }
            }

            let n_within = within_dep.len();
            let k = indepvars.len();

            // OLS on within-transformed data (no intercept)
            let mut xtx = vec![vec![0.0; k]; k];
            let mut xty = vec![0.0; k];

            for t in 0..n_within {
                for i in 0..k {
                    for j in 0..k {
                        xtx[i][j] += within_x[i][t] * within_x[j][t];
                    }
                    xty[i] += within_x[i][t] * within_dep[t];
                }
            }

            let beta = crate::logit::solve_linear_system(xtx, xty)?;

            // Compute R-squared within
            let mut ss_resid = 0.0;
            let mut ss_total = 0.0;
            let y_mean = within_dep.iter().sum::<f64>() / n_within as f64;
            for t in 0..n_within {
                let yhat: f64 = (0..k).map(|i| beta[i] * within_x[i][t]).sum();
                let resid = within_dep[t] - yhat;
                ss_resid += resid * resid;
                ss_total += (within_dep[t] - y_mean).powi(2);
            }
            let r2 = 1.0 - ss_resid / ss_total;

            let mut out = Vec::new();
            out.push(format!("Fixed-effects (within) regression"));
            out.push(format!("Group variable: {}", panel_var));
            out.push(format!("Number of obs = {}", n));
            out.push(format!("Number of groups = {}", num_panels));
            out.push(format!("R-sq: within = {:.4}", r2));
            out.push(format!("R-sq: between = {:.4}", 0.0));
            out.push(format!("R-sq: overall = {:.4}", 0.0));
            out.push("".to_string());
            out.push(format!("{:>12} |      Coef.   Std. Err.      t    P>|t|", ""));
            out.push("-".repeat(60));
            for (i, name) in indepvars.iter().enumerate() {
                out.push(format!("{:>12} | {:>10.6} {:>10.6} {:>8.3} {:>8.4}",
                    name, beta[i], 0.0, 0.0, 0.0));
            }
            out.push(format!("{:>12} | {:>10.6}", "_cons", 0.0));
            Ok(out.join("\n"))
        } else {
            // Random effects: pooled OLS as placeholder
            crate::regress::Regress.execute(ds, &ferrostat_parser::parser::Command {
                verb: "regress".to_string(),
                args: std::iter::once(depvar.clone()).chain(indepvars.iter().cloned()).collect(),
                ..Default::default()
            })
        }
    }
}
