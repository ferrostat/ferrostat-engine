use ferrostat_core::DataSet;
use ferrostat_parser::Command;
use crate::commands::CommandHandler;

/// Poisson regression.
pub struct Poisson;

impl CommandHandler for Poisson {
    fn name(&self) -> &'static str { "poisson" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.len() < 2 {
            return Err("poisson requires dependent variable and at least one independent variable".into());
        }

        let depvar = &cmd.args[0];
        let indepvars = &cmd.args[1..];

        let dep = ds.get_var(depvar)
            .ok_or_else(|| format!("variable {} not found", depvar))?
            .as_f64_vec()
            .ok_or("dependent variable must be numeric")?;

        let n = dep.len();
        let k = indepvars.len();
        let y: Vec<f64> = dep.iter().map(|&v| if v.is_nan() || v < 0.0 { -1.0 } else { v }).collect();

        let mut x: Vec<Vec<f64>> = vec![vec![1.0; n]];
        for name in indepvars {
            let col = ds.get_var(name)
                .ok_or_else(|| format!("variable {} not found", name))?
                .as_f64_vec()
                .ok_or("independent variable must be numeric")?;
            x.push(col);
        }

        let p = k + 1;

        // IRLS for Poisson regression with log link
        let mut beta = vec![0.0; p];
        for _iter in 0..100 {
            let mut xtx = vec![vec![0.0; p]; p];
            let mut xty = vec![0.0; p];

            for t in 0..n {
                if y[t] < 0.0 { continue; }
                let xb: f64 = (0..p).map(|i| beta[i] * x[i][t]).sum();
                let mu = xb.exp();
                let wi = mu; // weight for log link
                if wi < 1e-8 { continue; }
                let zi = xb + (y[t] - mu) / mu;

                for i in 0..p {
                    for j in 0..p {
                        xtx[i][j] += x[i][t] * x[j][t] * wi;
                    }
                    xty[i] += x[i][t] * wi * zi;
                }
            }

            let new_beta = crate::logit::solve_linear_system(xtx, xty)?;
            let max_diff: f64 = beta.iter().zip(&new_beta).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            beta = new_beta;
            if max_diff < 1e-6 { break; }
        }

        let mut ll = 0.0;
        for t in 0..n {
            if y[t] < 0.0 { continue; }
            let xb: f64 = (0..p).map(|i| beta[i] * x[i][t]).sum();
            let mu = xb.exp();
            // ln L = sum(y * ln(mu) - mu - ln(y!))
            // Stirling approx for ln(y!)
            let lnfact = if y[t] == 0.0 { 0.0 } else { y[t] * y[t].ln() - y[t] };
            ll += y[t] * xb - mu - lnfact;
        }

        let mut out = Vec::new();
        out.push(format!("Poisson regression"));
        out.push(format!("Number of obs = {}", n));
        out.push(format!("Log likelihood = {:.6}", ll));
        out.push("".to_string());
        out.push(format!("{:>12} |      Coef.   Std. Err.      z    P>|z|", ""));
        out.push("-".repeat(60));
        let mut names: Vec<String> = vec!["_cons".to_string()];
        names.extend(indepvars.iter().cloned());
        for (i, name) in names.iter().enumerate() {
            out.push(format!("{:>12} | {:>10.6} {:>10.6} {:>8.3} {:>8.4}",
                name, beta[i], 0.0, 0.0, 0.0));
        }
        Ok(out.join("\n"))
    }
}
