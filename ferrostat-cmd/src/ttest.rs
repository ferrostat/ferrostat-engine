use ferrostat_core::DataSet;
use ferrostat_parser::Command;
use crate::commands::CommandHandler;

/// T-test: one-sample, two-sample, and paired.
pub struct Ttest;

impl CommandHandler for Ttest {
    fn name(&self) -> &'static str { "ttest" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.len() < 2 {
            return Err("ttest requires at least a variable and a value or second variable".into());
        }

        let var1 = &cmd.args[0];
        let v1 = ds.get_var(var1)
            .ok_or_else(|| format!("variable {} not found", var1))?
            .as_f64_vec()
            .ok_or("variable must be numeric")?;

        let mut rows: Vec<usize> = (0..ds.n_obs).collect();
        if let Some(ref expr) = cmd.if_expr {
            rows = crate::summarize::apply_if(ds, expr)?;
        }

        let vals1: Vec<f64> = rows.iter().filter_map(|&i| v1.get(i)).copied().filter(|v| !v.is_nan()).collect();

        let n1 = vals1.len() as f64;
        if n1 < 2.0 {
            return Err("insufficient observations".into());
        }

        let mean1 = vals1.iter().sum::<f64>() / n1;
        let var1 = vals1.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
        let sd1 = var1.sqrt();

        let mut out = Vec::new();

        // Check if second arg is a constant or variable
        if let Ok(mu0) = cmd.args[1].parse::<f64>() {
            // One-sample t-test against mu0
            let se = (var1 / n1).sqrt();
            let t = (mean1 - mu0) / se;
            let df = n1 - 1.0;

            out.push(format!("One-sample t test"));
            out.push(format!("Variable |     Obs        Mean    Std. Err.   Std. Dev.   [95% Conf. Interval]"));
            out.push(format!("{:>8} | {:>7} {:>11.6} {:>11.6} {:>11.6} {:>11.6} {:>11.6}",
                var1, n1 as usize, mean1, se, sd1,
                mean1 - 1.96 * se, mean1 + 1.96 * se));
            out.push("".to_string());
            out.push(format!("    mean = {:>11.6} (mean of {})", mean1, var1));
            out.push(format!("    H0: mean = {}", mu0));
            out.push(format!("    t = {:>10.4}", t));
            out.push(format!("    df = {}", df as usize));

        } else if cmd.args.len() >= 2 {
            let var2 = &cmd.args[1];
            if let Some(v2_obj) = ds.get_var(var2) {
                let v2 = v2_obj.as_f64_vec().ok_or("second variable must be numeric")?;
                let vals2: Vec<f64> = rows.iter().filter_map(|&i| v2.get(i)).copied().filter(|v| !v.is_nan()).collect();
                let n2 = vals2.len() as f64;

                if n2 < 2.0 {
                    return Err("insufficient observations in second variable".into());
                }

                let mean2 = vals2.iter().sum::<f64>() / n2;
                let var2 = vals2.iter().map(|&x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);
                let sd2 = var2.sqrt();

                // Two-sample t-test (pooled)
                let pooled_var = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0);
                let se = (pooled_var * (1.0 / n1 + 1.0 / n2)).sqrt();
                let t = (mean1 - mean2) / se;
                let df = n1 + n2 - 2.0;

                out.push(format!("Two-sample t test with equal variances"));
                out.push(format!("Variable |     Obs        Mean    Std. Err.   Std. Dev.   [95% Conf. Interval]"));
                out.push(format!("{:>8} | {:>7} {:>11.6} {:>11.6} {:>11.6} {:>11.6} {:>11.6}",
                    var1, n1 as usize, mean1, (var1/n1).sqrt(), sd1,
                    mean1 - 1.96 * (var1/n1).sqrt(), mean1 + 1.96 * (var1/n1).sqrt()));
                out.push(format!("{:>8} | {:>7} {:>11.6} {:>11.6} {:>11.6} {:>11.6} {:>11.6}",
                    var2, n2 as usize, mean2, (var2/n2).sqrt(), sd2,
                    mean2 - 1.96 * (var2/n2).sqrt(), mean2 + 1.96 * (var2/n2).sqrt()));
                out.push(format!("combined | {:>7} {:>11.6} {:>11.6} {:>11.6}",
                    (n1 + n2) as usize,
                    (n1 * mean1 + n2 * mean2) / (n1 + n2),
                    (pooled_var / (n1 + n2)).sqrt(),
                    pooled_var.sqrt()));
                out.push("".to_string());
                out.push(format!("    diff = {:>10.6}", mean1 - mean2));
                out.push(format!("    t = {:>10.4}", t));
                out.push(format!("    df = {}", df as usize));
            } else {
                return Err(format!("variable {} not found", var2));
            }
        }

        Ok(out.join("\n"))
    }
}
