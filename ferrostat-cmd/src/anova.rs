use ferrostat_core::DataSet;
use ferrostat_parser::Command;
use crate::commands::CommandHandler;

/// One-way ANOVA.
pub struct Anova;

impl CommandHandler for Anova {
    fn name(&self) -> &'static str { "anova" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.len() < 2 {
            return Err("anova requires a dependent variable and at least one factor".into());
        }

        let depvar = &cmd.args[0];
        let factor = &cmd.args[1];

        let dep = ds.get_var(depvar)
            .ok_or_else(|| format!("variable {} not found", depvar))?
            .as_f64_vec()
            .ok_or("dependent variable must be numeric")?;

        let n = dep.len();

        // Group by factor
        let mut groups: Vec<Vec<f64>> = Vec::new();
        let mut group_labels: Vec<String> = Vec::new();

        let factor_var = ds.get_var(factor).ok_or_else(|| format!("variable {} not found", factor))?;

        for i in 0..n {
            let label = match &factor_var.data[i] {
                ferrostat_core::Value::Str(s) => s.clone(),
                ferrostat_core::Value::Int(v) => v.to_string(),
                ferrostat_core::Value::Long(v) => v.to_string(),
                ferrostat_core::Value::Byte(v) => v.to_string(),
                ferrostat_core::Value::Float(v) => format!("{:.0}", v),
                ferrostat_core::Value::Double(v) => format!("{:.0}", v),
                _ => continue,
            };

            if let Some(pos) = group_labels.iter().position(|x| x == &label) {
                if let Some(val) = dep.get(i) {
                    if !val.is_nan() {
                        groups[pos].push(*val);
                    }
                }
            } else {
                group_labels.push(label);
                let mut new_group = Vec::new();
                if let Some(val) = dep.get(i) {
                    if !val.is_nan() {
                        new_group.push(*val);
                    }
                }
                groups.push(new_group);
            }
        }

        let k = groups.len();
        let grand_mean: f64 = dep.iter().filter(|&&v| !v.is_nan()).sum::<f64>()
            / dep.iter().filter(|&&v| !v.is_nan()).count() as f64;

        let mut ss_between = 0.0;
        let mut ss_within = 0.0;
        let mut total_n = 0usize;

        for group in &groups {
            let n_i = group.len() as f64;
            let mean_i = group.iter().sum::<f64>() / n_i;
            ss_between += n_i * (mean_i - grand_mean).powi(2);
            for &x in group {
                ss_within += (x - mean_i).powi(2);
            }
            total_n += group.len();
        }

        let ss_total = ss_between + ss_within;
        let df_between = k - 1;
        let df_within = total_n - k;
        let df_total = total_n - 1;

        let ms_between = ss_between / df_between as f64;
        let ms_within = ss_within / df_within as f64;
        let f_stat = if ms_within > 0.0 { ms_between / ms_within } else { 0.0 };

        let mut out = Vec::new();
        out.push(format!("                   Number of obs = {}", total_n));
        out.push(format!("{}-way ANOVA with {} categories", 1, k));
        out.push(format!("Root MSE = {:.4}", ms_within.sqrt()));
        out.push("".to_string());
        out.push(format!("Source |       SS           df       MS            F     Prob > F"));
        out.push("-".repeat(65));
        out.push(format!("{:>6} | {:>12.4} {:>10} {:>12.4} {:>12.4} {:>12.4}",
            "Between", ss_between, df_between, ms_between, f_stat, 0.0));
        out.push(format!("{:>6} | {:>12.4} {:>10} {:>12.4}",
            "Within", ss_within, df_within, ms_within));
        out.push(format!("{:>6} | {:>12.4} {:>10} {:>12.4}",
            "Total", ss_total, df_total, ss_total / df_total as f64));
        out.push("".to_string());
        out.push(format!("{:>12} |      Coef.   Std. Err.      t    P>|t|   [95% Conf. Interval]", ""));
        out.push("-".repeat(72));
        out.push(format!("{:>12} | {:>10.6} {:>10.6} {:>8.3} {:>8.4}",
            "_cons", grand_mean, ms_within.sqrt(), grand_mean / ms_within.sqrt(), 0.0));

        Ok(out.join("\n"))
    }
}
