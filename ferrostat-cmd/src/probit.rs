use ferrostat_core::DataSet;
use ferrostat_parser::Command;
use crate::commands::CommandHandler;

/// Probit regression.
pub struct Probit;

impl CommandHandler for Probit {
    fn name(&self) -> &'static str { "probit" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.len() < 2 {
            return Err("probit requires dependent variable and at least one independent variable".into());
        }

        let depvar = &cmd.args[0];
        let indepvars = &cmd.args[1..];

        let dep = ds.get_var(depvar)
            .ok_or_else(|| format!("variable {} not found", depvar))?
            .as_f64_vec()
            .ok_or("dependent variable must be numeric")?;

        let n = dep.len();
        let k = indepvars.len();
        let y: Vec<f64> = dep.iter().map(|&v| if v.is_nan() { -1.0 } else { v }).collect();

        let mut x: Vec<Vec<f64>> = vec![vec![1.0; n]];
        for name in indepvars {
            let col = ds.get_var(name)
                .ok_or_else(|| format!("variable {} not found", name))?
                .as_f64_vec()
                .ok_or("independent variable must be numeric")?;
            x.push(col);
        }

        let p = k + 1;

        // IRLS for probit regression
        let mut beta = vec![0.0; p];
        for _iter in 0..100 {
            let mut xtx = vec![vec![0.0; p]; p];
            let mut xty = vec![0.0; p];

            for t in 0..n {
                if y[t] < 0.0 { continue; }
                let xb: f64 = (0..p).map(|i| beta[i] * x[i][t]).sum();
                // phi = normal pdf, Phi = normal cdf
                let phi_val = (2.0 * std::f64::consts::PI).sqrt().recip() * (-xb * xb / 2.0).exp();
                let phi_cdf = 0.5 * (1.0 + erf_approx(xb / 2.0f64.sqrt()));
                let pi = phi_cdf;
                let mi = if pi > 0.9999 { 0.0001 } else if pi < 0.0001 { 0.9999 } else { phi_val / (pi * (1.0 - pi)) };
                let wi = phi_val * mi;
                if wi < 1e-8 { continue; }
                let zi = xb + (y[t] - pi) * mi;

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
            let pi = 0.5 * (1.0 + erf_approx(xb / 2.0f64.sqrt()));
            ll += y[t] * pi.ln() + (1.0 - y[t]) * (1.0 - pi).ln();
        }

        let mut out = Vec::new();
        out.push(format!("Probit regression"));
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

fn erf_approx(x: f64) -> f64 {
    // Abramowitz & Stegun approximation
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}
