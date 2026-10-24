use std::collections::HashMap;

use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::Command;
use crate::commands::CommandHandler;

/// Collapse: aggregate statistics.
pub struct Collapse;

impl CommandHandler for Collapse {
    fn name(&self) -> &'static str { "collapse" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.is_empty() {
            return Err("collapse requires statistics and variables".into());
        }

        // Simple parsing: <stat> <varlist> ...
        // For now, support: (mean) var1 var2, (sum) var3, (count), (max) var, (min) var
        let mut stat_vars: Vec<(String, Vec<String>)> = Vec::new();
        let mut current_stat = "mean".to_string();
        let mut current_vars: Vec<String> = Vec::new();

        for arg in &cmd.args {
            if arg.starts_with('(') && arg.ends_with(')') {
                if !current_vars.is_empty() {
                    stat_vars.push((current_stat.clone(), current_vars.clone()));
                    current_vars.clear();
                }
                current_stat = arg[1..arg.len()-1].to_string();
            } else {
                current_vars.push(arg.clone());
            }
        }
        if !current_vars.is_empty() {
            stat_vars.push((current_stat, current_vars));
        }

        // Get by-variables if specified
        let by_vars: Vec<String> = cmd.options.iter()
            .filter(|(k, _)| k == "by")
            .filter_map(|(_, v)| v.as_ref().and_then(crate::eval::expr_to_string))
            .collect();

        // Aggregate
        if by_vars.is_empty() {
            // Simple aggregation to one row
            let mut new_ds = DataSet::new();
            for (stat, vars) in &stat_vars {
                for var in vars {
                    let v = ds.get_var(var).ok_or_else(|| format!("variable {} not found", var))?;
                    let vals: Vec<f64> = v.data.iter().filter_map(|val| val.as_f64()).filter(|x| !x.is_nan()).collect();
                    let n = vals.len() as f64;
                    if n == 0.0 { continue; }

                    let result = match stat.as_str() {
                        "mean" => vals.iter().sum::<f64>() / n,
                        "sum" => vals.iter().sum(),
                        "count" => n,
                        "max" => vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                        "min" => vals.iter().cloned().fold(f64::INFINITY, f64::min),
                        "sd" => {
                            let mean = vals.iter().sum::<f64>() / n;
                            (vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
                        }
                        _ => vals.iter().sum::<f64>() / n,
                    };

                    let mut new_var = Variable::new(format!("{}_{}", stat, var), VarType::Double);
                    new_var.data.push(Value::Double(result));
                    let _ = new_ds.add_variable(new_var);
                }
            }
            new_ds.n_obs = 1;
            *ds = new_ds;
        } else {
            // Group by
            let mut groups: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
            for i in 0..ds.n_obs {
                let mut key = Vec::new();
                for bv in &by_vars {
                    let val_str = match &ds.get_var(bv).ok_or("variable not found")?.data[i] {
                        Value::Str(s) => s.clone(),
                        Value::Double(v) => format!("{}", v),
                        Value::Float(v) => format!("{}", v),
                        Value::Int(v) => v.to_string(),
                        Value::Long(v) => v.to_string(),
                        Value::Byte(v) => v.to_string(),
                        _ => "".to_string(),
                    };
                    key.push(val_str);
                }
                groups.entry(key).or_insert_with(Vec::new).push(i);
            }

            let mut new_ds = DataSet::new();
            // Add by variables
            for bv in &by_vars {
                let old = ds.get_var(bv).ok_or("variable not found")?;
                let nv = Variable::new(bv.clone(), old.var_type);
                let _ = new_ds.add_variable(nv);
            }
            // Add result variables
            for (stat, vars) in &stat_vars {
                for var in vars {
                    let nv = Variable::new(format!("{}_{}", stat, var), VarType::Double);
                    let _ = new_ds.add_variable(nv);
                }
            }

            for (_key, indices) in &groups {
                for (j, bv) in by_vars.iter().enumerate() {
                    let old = ds.get_var(bv).ok_or("variable not found")?;
                    new_ds.vars[j].data.push(old.data[indices[0]].clone());
                }

                let mut col = by_vars.len();
                for (stat, vars) in &stat_vars {
                    for var in vars {
                        let v = ds.get_var(var).ok_or_else(|| format!("variable {} not found", var))?;
                        let vals: Vec<f64> = indices.iter().filter_map(|&i| v.data[i].as_f64()).filter(|x| !x.is_nan()).collect();
                        let n = vals.len() as f64;
                        let result = if n == 0.0 {
                            f64::NAN
                        } else {
                            match stat.as_str() {
                                "mean" => vals.iter().sum::<f64>() / n,
                                "sum" => vals.iter().sum(),
                                "count" => n,
                                "max" => vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                                "min" => vals.iter().cloned().fold(f64::INFINITY, f64::min),
                                "sd" => {
                                    let mean = vals.iter().sum::<f64>() / n;
                                    (vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
                                }
                                _ => vals.iter().sum::<f64>() / n,
                            }
                        };
                        new_ds.vars[col].data.push(Value::Double(result));
                        col += 1;
                    }
                }
            }
            new_ds.n_obs = new_ds.vars[0].data.len();
            *ds = new_ds;
        }

        Ok("data collapsed".to_string())
    }
}
