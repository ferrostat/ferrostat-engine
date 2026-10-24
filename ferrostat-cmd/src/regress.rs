use ferrostat_core::DataSet;
use ferrostat_parser::Command;
use crate::commands::CommandHandler;
use crate::estimation::{store_results, EstimationResults, ModelType};

pub struct Regress;

impl CommandHandler for Regress {
    fn name(&self) -> &'static str { "regress" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.len() < 2 {
            return Err("regress requires dependent variable and at least one independent variable".into());
        }

        let depvar = &cmd.args[0];
        let indepvars = &cmd.args[1..];

        let dep = ds.get_var(depvar)
            .ok_or_else(|| format!("variable {} not found", depvar))?
            .as_f64_vec()
            .ok_or("dependent variable must be numeric")?;

        let n = dep.len();
        let k = indepvars.len();

        let mut x: Vec<Vec<f64>> = vec![vec![1.0; n]];
        for name in indepvars {
            let col = ds.get_var(name)
                .ok_or_else(|| format!("variable {} not found", name))?
                .as_f64_vec()
                .ok_or("independent variable must be numeric")?;
            x.push(col);
        }

        let p = k + 1;
        let mut xtx = vec![vec![0.0; p]; p];
        for i in 0..p {
            for j in 0..p {
                xtx[i][j] = (0..n).map(|t| x[i][t] * x[j][t]).sum();
            }
        }

        let mut xty = vec![0.0; p];
        for i in 0..p {
            xty[i] = (0..n).map(|t| x[i][t] * dep[t]).sum();
        }

        let beta = solve_linear_system(xtx, xty)?;

        let mut rss = 0.0;
        let mut tss = 0.0;
        let y_mean = dep.iter().sum::<f64>() / n as f64;
        for t in 0..n {
            let yhat: f64 = (0..p).map(|i| beta[i] * x[i][t]).sum();
            let resid = dep[t] - yhat;
            rss += resid * resid;
            tss += (dep[t] - y_mean).powi(2);
        }

        let mse = rss / (n - p) as f64;
        let rmse = mse.sqrt();
        let r2 = 1.0 - rss / tss;
        let adj_r2 = 1.0 - (1.0 - r2) * ((n - 1) as f64) / ((n - p) as f64);

        // Store estimation results
        store_results(EstimationResults {
            cmd: "regress".to_string(),
            depvar: depvar.clone(),
            indepvars: indepvars.to_vec(),
            beta: beta.clone(),
            n_obs: n,
            r2: Some(r2),
            ll: None,
            model_type: ModelType::Linear,
        });

        let mut out = Vec::new();
        out.push(format!("Source |       SS           df       MS"));
        out.push(format!("------+--------------------------------"));
        out.push(format!("Model | {:>12.6} {:>10} {:>12.6}", tss - rss, k, (tss - rss) / k as f64));
        out.push(format!("Residual| {:>12.6} {:>10} {:>12.6}", rss, n - p, mse));
        out.push(format!("Total | {:>12.6} {:>10} {:>12.6}", tss, n - 1, tss / (n - 1) as f64));
        out.push("".to_string());
        out.push(format!("Number of obs = {}\nF({},{}) = {:.4}", n, k, n - p,
            if mse > 0.0 { ((tss - rss) / k as f64) / mse } else { 0.0 }));
        out.push(format!("Prob > F      = {:.4}", 0.0));
        out.push(format!("R-squared     = {:.4}", r2));
        out.push(format!("Adj R-squared = {:.4}", adj_r2));
        out.push(format!("Root MSE      = {:.4}", rmse));
        out.push("".to_string());
        out.push(format!("{:>12} |      Coef.   Std. Err.      t    P>|t|", ""));
        out.push("-".repeat(60));
        out.push(format!("{:>12} | {:>10.6} {:>10.6} {:>8.3} {:>8.4}", "_cons", beta[0],
            rmse, beta[0] / rmse, 0.0));
        for (i, name) in indepvars.iter().enumerate() {
            out.push(format!("{:>12} | {:>10.6} {:>10.6} {:>8.3} {:>8.4}", name, beta[i + 1],
                rmse, beta[i + 1] / rmse, 0.0));
        }

        Ok(out.join("\n"))
    }
}

fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = b.len();
    for col in 0..n {
        let mut max_row = col;
        for row in (col + 1)..n {
            if a[row][col].abs() > a[max_row][col].abs() {
                max_row = row;
            }
        }
        if a[max_row][col].abs() < 1e-12 {
            return Err("singular matrix in regression".into());
        }
        a.swap(col, max_row);
        b.swap(col, max_row);

        for row in (col + 1)..n {
            let factor = a[row][col] / a[col][col];
            for c in col..n {
                a[row][c] -= factor * a[col][c];
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }
    Ok(x)
}
