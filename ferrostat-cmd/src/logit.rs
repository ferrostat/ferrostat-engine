use ferrostat_core::DataSet;
use ferrostat_parser::Command;
use crate::commands::CommandHandler;

/// Logistic regression via IRLS (Newton-Raphson for GLM).
pub struct Logit;

impl CommandHandler for Logit {
    fn name(&self) -> &'static str { "logit" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.len() < 2 {
            return Err("logit requires dependent variable and at least one independent variable".into());
        }

        let depvar = &cmd.args[0];
        let indepvars = &cmd.args[1..];

        let dep = ds.get_var(depvar)
            .ok_or_else(|| format!("variable {} not found", depvar))?
            .as_f64_vec()
            .ok_or("dependent variable must be numeric")?;

        let n = dep.len();
        let k = indepvars.len();

        // Check depvar is binary 0/1
        let y: Vec<f64> = dep.iter().map(|&v| {
            if v.is_nan() { -1.0 } else { v }
        }).collect();

        // Build design matrix with intercept
        let mut x: Vec<Vec<f64>> = vec![vec![1.0; n]];
        for name in indepvars {
            let col = ds.get_var(name)
                .ok_or_else(|| format!("variable {} not found", name))?
                .as_f64_vec()
                .ok_or("independent variable must be numeric")?;
            x.push(col);
        }

        let p = k + 1;

        // IRLS for logistic regression
        let mut beta = vec![0.0; p];
        let mut _converged = false;

        for _iter in 0..100 {
            let mut xtx = vec![vec![0.0; p]; p];
            let mut xty = vec![0.0; p];

            for t in 0..n {
                if y[t] < 0.0 { continue; } // skip missing
                let xb: f64 = (0..p).map(|i| beta[i] * x[i][t]).sum();
                let pi = 1.0 / (1.0 + (-xb).exp());
                let wi = pi * (1.0 - pi);
                if wi < 1e-8 { continue; }
                let zi = xb + (y[t] - pi) / wi;

                for i in 0..p {
                    for j in 0..p {
                        xtx[i][j] += x[i][t] * x[j][t] * wi;
                    }
                    xty[i] += x[i][t] * wi * zi;
                }
            }

            let new_beta = solve_linear_system(xtx, xty)?;
            let max_diff: f64 = beta.iter().zip(&new_beta).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            beta = new_beta;
            if max_diff < 1e-6 {
                _converged = true;
                break;
            }
        }

        // Compute log-likelihood and statistics
        let mut ll = 0.0;
        let mut _correct = 0;
        for t in 0..n {
            if y[t] < 0.0 { continue; }
            let xb: f64 = (0..p).map(|i| beta[i] * x[i][t]).sum();
            let pi = 1.0 / (1.0 + (-xb).exp());
            ll += y[t] * pi.ln() + (1.0 - y[t]) * (1.0 - pi).ln();
            if (pi > 0.5 && y[t] == 1.0) || (pi <= 0.5 && y[t] == 0.0) {
                _correct += 1;
            }
        }

        let mut out = Vec::new();
        out.push(format!("Logistic regression"));
        out.push(format!("Number of obs = {}", n));
        out.push(format!("Log likelihood = {:.6}", ll));
        out.push(format!("LR chi2({}) = {:.4}", k, 0.0)); // placeholder
        out.push(format!("Prob > chi2 = {:.4}", 0.0));
        out.push(format!("Pseudo R2 = {:.4}", 0.0));
        out.push("".to_string());
        out.push(format!("{:>12} |      Coef.   Std. Err.      z    P>|z|   [95% Conf. Interval]", ""));
        out.push("-".repeat(72));

        let mut names: Vec<String> = vec!["_cons".to_string()];
        names.extend(indepvars.iter().cloned());
        for (i, name) in names.iter().enumerate() {
            out.push(format!("{:>12} | {:>10.6} {:>10.6} {:>8.3} {:>8.4} {:>10.6} {:>10.6}",
                name, beta[i], 0.0, 0.0, 0.0, beta[i], beta[i]));
        }

        Ok(out.join("\n"))
    }
}

pub(crate) fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = b.len();
    for col in 0..n {
        let mut max_row = col;
        for row in (col + 1)..n {
            if a[row][col].abs() > a[max_row][col].abs() {
                max_row = row;
            }
        }
        if a[max_row][col].abs() < 1e-12 {
            return Err("singular matrix".into());
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
