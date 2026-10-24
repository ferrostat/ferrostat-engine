use std::f64::consts::PI;

use ferrostat_core::DataSet;
use ferrostat_parser::Command;
use crate::commands::CommandHandler;

/// Kernel density estimation.
pub struct Kdensity;

impl CommandHandler for Kdensity {
    fn name(&self) -> &'static str { "kdensity" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.is_empty() {
            return Err("kdensity requires a variable".into());
        }

        let var = &cmd.args[0];
        let v = ds.get_var(var)
            .ok_or_else(|| format!("variable {} not found", var))?
            .as_f64_vec()
            .ok_or("variable must be numeric")?;

        let mut rows: Vec<usize> = (0..ds.n_obs).collect();
        if let Some(ref expr) = cmd.if_expr {
            rows = crate::summarize::apply_if(ds, expr)?;
        }
        if let Some(ref range) = cmd.in_range {
            rows = crate::summarize::apply_in(ds, range)?;
        }

        let vals: Vec<f64> = rows.iter()
            .filter_map(|&i| v.get(i))
            .copied()
            .filter(|x| !x.is_nan())
            .collect();

        let n = vals.len() as f64;
        if n < 2.0 {
            return Err("insufficient observations".into());
        }

        let mean = vals.iter().sum::<f64>() / n;
        let var = vals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let sd = var.sqrt();
        let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Silverman's rule of thumb bandwidth
        let iqr = percentile(&vals, 0.75) - percentile(&vals, 0.25);
        let h = 0.9 * f64::min(sd, iqr / 1.34) * n.powf(-0.2);

        // Generate density points
        let n_points = 50;
        let mut out = Vec::new();
        out.push(format!("Kernel density estimation"));
        out.push(format!("Variable: {}", var));
        out.push(format!("Observations: {}", n as usize));
        out.push(format!("Bandwidth: {:.6}", h));
        out.push(format!("       x               density"));
        out.push("-".repeat(40));

        for i in 0..=n_points {
            let x = min + (max - min) * (i as f64 / n_points as f64);
            let mut density = 0.0;
            for &xi in &vals {
                let u = (x - xi) / h;
                density += gaussian_kernel(u);
            }
            density /= n * h;
            out.push(format!("{:>11.6} {:>18.6}", x, density));
        }

        Ok(out.join("\n"))
    }
}

fn gaussian_kernel(u: f64) -> f64 {
    (2.0 * PI).sqrt().recip() * (-0.5 * u * u).exp()
}

fn percentile(data: &[f64], p: f64) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
