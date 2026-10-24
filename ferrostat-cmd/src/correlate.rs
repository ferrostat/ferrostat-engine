use ferrostat_core::DataSet;
use ferrostat_parser::Command;
use crate::commands::CommandHandler;

/// Correlation matrix.
pub struct Correlate;

impl CommandHandler for Correlate {
    fn name(&self) -> &'static str { "correlate" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        let varlist: Vec<String> = if cmd.args.is_empty() {
            ds.varlist().into_iter().filter(|n| {
                ds.get_var(n).map(|v| v.var_type.is_numeric()).unwrap_or(false)
            }).collect()
        } else {
            cmd.args.clone()
        };

        if varlist.len() < 2 {
            return Err("correlate requires at least 2 variables".into());
        }

        let mut selected: Vec<usize> = (0..ds.n_obs).collect();
        if let Some(ref expr) = cmd.if_expr {
            selected = crate::summarize::apply_if(ds, expr)?;
        }
        if let Some(ref range) = cmd.in_range {
            selected = crate::summarize::apply_in(ds, range)?;
        }

        // Collect numeric vectors
        let mut vecs = Vec::new();
        for name in &varlist {
            let var = ds.get_var(name).ok_or_else(|| format!("variable {} not found", name))?;
            let vals: Vec<f64> = selected.iter()
                .filter_map(|&i| var.data[i].as_f64())
                .filter(|v| !v.is_nan())
                .collect();
            vecs.push(vals);
        }

        let n = vecs[0].len();
        let k = varlist.len();

        // Compute means
        let means: Vec<f64> = vecs.iter().map(|v| v.iter().sum::<f64>() / n as f64).collect();

        // Compute standard deviations
        let stds: Vec<f64> = vecs.iter().zip(&means)
            .map(|(v, &m)| {
                let var = v.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (n as f64 - 1.0);
                var.sqrt()
            })
            .collect();

        // Compute correlation matrix
        let mut corr = vec![vec![0.0; k]; k];
        for i in 0..k {
            for j in 0..k {
                if i == j {
                    corr[i][j] = 1.0;
                } else {
                    let cov: f64 = vecs[i].iter().zip(&vecs[j])
                        .map(|(&xi, &xj)| (xi - means[i]) * (xj - means[j]))
                        .sum::<f64>() / (n as f64 - 1.0);
                    if stds[i] > 0.0 && stds[j] > 0.0 {
                        corr[i][j] = cov / (stds[i] * stds[j]);
                    }
                }
            }
        }

        let mut out = Vec::new();
        let max_name_len = varlist.iter().map(|s| s.len()).max().unwrap_or(10);
        let mut header = format!("{:>width$} |", "", width = max_name_len);
        for name in &varlist {
            header.push_str(&format!(" {:>8}", &name[..name.len().min(8)]));
        }
        out.push(header);
        out.push("-".repeat(max_name_len + 2 + k * 9));

        for i in 0..k {
            let mut line = format!("{:>width$} |", varlist[i], width = max_name_len);
            for j in 0..k {
                line.push_str(&format!(" {:>8.4}", corr[i][j]));
            }
            out.push(line);
        }

        out.push("".to_string());
        out.push(format!("obs = {}", n));
        Ok(out.join("\n"))
    }
}
