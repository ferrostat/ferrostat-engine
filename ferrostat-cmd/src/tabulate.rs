use std::collections::HashMap;
use ferrostat_core::{DataSet, Value};
use ferrostat_parser::Command;
use crate::commands::CommandHandler;

/// Tabulate: frequency tables and cross-tabulations.
pub struct Tabulate;

impl CommandHandler for Tabulate {
    fn name(&self) -> &'static str { "tabulate" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.is_empty() {
            return Err("tabulate requires at least one variable".into());
        }

        let var1 = &cmd.args[0];
        let _v1 = ds.get_var(var1).ok_or_else(|| format!("variable {} not found", var1))?;

        if cmd.args.len() == 1 {
            // One-way table
            self.one_way(ds, var1, cmd)
        } else {
            // Two-way table
            let var2 = &cmd.args[1];
            self.two_way(ds, var1, var2, cmd)
        }
    }
}

impl Tabulate {
    fn one_way(&self, ds: &DataSet, var: &str, cmd: &Command) -> Result<String, String> {
        let mut rows: Vec<usize> = (0..ds.n_obs).collect();
        if let Some(ref expr) = cmd.if_expr {
            rows = crate::summarize::apply_if(ds, expr)?;
        }
        if let Some(ref range) = cmd.in_range {
            rows = crate::summarize::apply_in(ds, range)?;
        }

        let var_obj = ds.get_var(var).unwrap();
        let mut counts: HashMap<String, usize> = HashMap::new();
        for &i in &rows {
            let key = value_to_string(&var_obj.data[i], ds);
            *counts.entry(key).or_insert(0) += 1;
        }

        let total = rows.len() as f64;
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut out = Vec::new();
        out.push(format!("{} |      Freq.     Percent        Cum.", var));
        out.push("-".repeat(50));
        let mut cum = 0.0;
        for (label, count) in sorted {
            let pct = (count as f64 / total) * 100.0;
            cum += pct;
            out.push(format!("{:>12} | {:>10} {:>10.2} {:>10.2}", label, count, pct, cum));
        }
        out.push("-".repeat(50));
        out.push(format!("       Total | {:>10} {:>10.2}", rows.len(), 100.0));
        Ok(out.join("\n"))
    }

    fn two_way(&self, ds: &DataSet, var1: &str, var2: &str, cmd: &Command) -> Result<String, String> {
        let mut rows: Vec<usize> = (0..ds.n_obs).collect();
        if let Some(ref expr) = cmd.if_expr {
            rows = crate::summarize::apply_if(ds, expr)?;
        }
        if let Some(ref range) = cmd.in_range {
            rows = crate::summarize::apply_in(ds, range)?;
        }

        let v1 = ds.get_var(var1).unwrap();
        let v2 = ds.get_var(var2).unwrap();

        let mut table: HashMap<(String, String), usize> = HashMap::new();
        let mut row_totals: HashMap<String, usize> = HashMap::new();
        let mut col_totals: HashMap<String, usize> = HashMap::new();

        for &i in &rows {
            let key1 = value_to_string(&v1.data[i], ds);
            let key2 = value_to_string(&v2.data[i], ds);
            *table.entry((key1.clone(), key2.clone())).or_insert(0) += 1;
            *row_totals.entry(key1).or_insert(0) += 1;
            *col_totals.entry(key2).or_insert(0) += 1;
        }

        let mut cols: Vec<_> = col_totals.keys().cloned().collect();
        cols.sort();
        let mut rows_keys: Vec<_> = row_totals.keys().cloned().collect();
        rows_keys.sort();

        let total = rows.len();

        let mut out = Vec::new();
        let header = format!("{:>12} | {}", var1, cols.iter().map(|c| format!("{:>10}", c)).collect::<Vec<_>>().join(" "));
        out.push(header);
        out.push("-".repeat(14 + cols.len() * 11));

        for r in &rows_keys {
            let mut parts = vec![format!("{:>12}", r)];
            for c in &cols {
                let count = table.get(&(r.clone(), c.clone())).copied().unwrap_or(0);
                parts.push(format!("{:>10}", count));
            }
            out.push(parts.join(" | "));
        }
        out.push("-".repeat(14 + cols.len() * 11));

        // Column totals
        let mut parts = vec![format!("{:>12}", "Total")];
        for c in &cols {
            parts.push(format!("{:>10}", col_totals.get(c).copied().unwrap_or(0)));
        }
        out.push(parts.join(" | "));

        // Chi-square test (simplified)
        if cols.len() > 1 && rows_keys.len() > 1 {
            let mut chi2 = 0.0;
            for r in &rows_keys {
                for c in &cols {
                    let observed = *table.get(&(r.clone(), c.clone())).unwrap_or(&0) as f64;
                    let expected = (*row_totals.get(r).unwrap_or(&0) as f64)
                        * (*col_totals.get(c).unwrap_or(&0) as f64)
                        / (total as f64);
                    if expected > 0.0 {
                        chi2 += (observed - expected).powi(2) / expected;
                    }
                }
            }
            let df = (rows_keys.len() - 1) * (cols.len() - 1);
            out.push("".to_string());
            out.push(format!("Pearson chi2({}) = {:.4}", df, chi2));
        }

        Ok(out.join("\n"))
    }
}

fn value_to_string(val: &Value, ds: &DataSet) -> String {
    match val {
        Value::Str(s) => s.clone(),
        Value::Double(v) => if v.is_nan() { ".".to_string() } else { format!("{:.4}", v) },
        Value::Float(v) => if v.is_nan() { ".".to_string() } else { format!("{:.4}", v) },
        Value::Int(v) => v.to_string(),
        Value::Long(v) => v.to_string(),
        Value::Byte(v) => v.to_string(),
        Value::StrL(v, o) => ds.resolve_strl(*v, *o).unwrap_or("(strL)").to_string(),
        Value::Missing => ".".to_string(),
    }
}
