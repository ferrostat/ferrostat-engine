use std::collections::HashMap;
use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::Command;
use crate::commands::CommandHandler;

/// Extended generate: group statistics, seq, rank, etc.
pub struct Egen;

impl CommandHandler for Egen {
    fn name(&self) -> &'static str { "egen" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        // Parse: egen newvar = fcn(arguments)
        // The raw text contains the full expression which may include parentheses
        let raw = &cmd.raw;
        let parts: Vec<&str> = raw.splitn(3, '=').collect();
        if parts.len() < 2 {
            return Err("egen syntax: egen newvar = fcn(arguments)".into());
        }
        
        let left = parts[0].trim();
        let newvar = left.split_whitespace().nth(1)
            .ok_or("egen requires a new variable name")?;
        
        if ds.get_var(newvar).is_some() {
            return Err(format!("variable {} already defined", newvar));
        }
        
        let right = parts[1..].join("=").trim().to_string();
        
        // Parse function name and arguments from right side
        let func_name: String;
        let mut func_args: Vec<String> = Vec::new();
        
        if let Some(open_pos) = right.find('(') {
            func_name = right[..open_pos].trim().to_lowercase();
            let close_pos = right.rfind(')').unwrap_or(right.len());
            let inner = &right[open_pos+1..close_pos];
            func_args = inner.split_whitespace().map(|s| s.trim_matches(',').to_string()).filter(|s| !s.is_empty()).collect();
        } else {
            func_name = right.to_lowercase();
        }

        let result = match func_name.as_str() {
            "rowmean" => egen_rowmean(ds, &func_args),
            "rowsum" => egen_rowsum(ds, &func_args),
            "rowtotal" => egen_rowsum(ds, &func_args),
            "rowmax" => egen_rowmax(ds, &func_args),
            "rowmin" => egen_rowmin(ds, &func_args),
            "rowmiss" => egen_rowmiss(ds, &func_args),
            "rownonmiss" => egen_rownonmiss(ds, &func_args),
            "group" => egen_group(ds, &func_args),
            "seq" => egen_seq(ds, &func_args),
            "rank" => egen_rank(ds, &func_args),
            "count" => egen_count(ds, &func_args),
            "mean" => egen_mean(ds, &func_args),
            "sd" => egen_sd(ds, &func_args),
            "max" => egen_max(ds, &func_args),
            "min" => egen_min(ds, &func_args),
            "total" => egen_total(ds, &func_args),
            _ => Err(format!("unknown egen function: {}", func_name)),
        };

        let mut var = Variable::new(newvar.clone(), VarType::Double);
        var.data = result?;
        ds.add_variable(var)?;
        Ok(format!("generated variable: {}", newvar))
    }
}

fn egen_rowmean(ds: &DataSet, vars: &[String]) -> Result<Vec<Value>, String> {
    let mut result = Vec::with_capacity(ds.n_obs);
    for i in 0..ds.n_obs {
        let mut sum = 0.0;
        let mut n = 0;
        for name in vars {
            if let Some(v) = ds.get_var(name) {
                if let Some(val) = v.data[i].as_f64() {
                    if !val.is_nan() {
                        sum += val;
                        n += 1;
                    }
                }
            }
        }
        result.push(Value::Double(if n > 0 { sum / n as f64 } else { f64::NAN }));
    }
    Ok(result)
}

fn egen_rowsum(ds: &DataSet, vars: &[String]) -> Result<Vec<Value>, String> {
    let mut result = Vec::with_capacity(ds.n_obs);
    for i in 0..ds.n_obs {
        let mut sum = 0.0;
        for name in vars {
            if let Some(v) = ds.get_var(name) {
                if let Some(val) = v.data[i].as_f64() {
                    if !val.is_nan() {
                        sum += val;
                    }
                }
            }
        }
        result.push(Value::Double(sum));
    }
    Ok(result)
}

fn egen_rowmax(ds: &DataSet, vars: &[String]) -> Result<Vec<Value>, String> {
    let mut result = Vec::with_capacity(ds.n_obs);
    for i in 0..ds.n_obs {
        let mut max = f64::NEG_INFINITY;
        let mut found = false;
        for name in vars {
            if let Some(v) = ds.get_var(name) {
                if let Some(val) = v.data[i].as_f64() {
                    if !val.is_nan() {
                        max = max.max(val);
                        found = true;
                    }
                }
            }
        }
        result.push(Value::Double(if found { max } else { f64::NAN }));
    }
    Ok(result)
}

fn egen_rowmin(ds: &DataSet, vars: &[String]) -> Result<Vec<Value>, String> {
    let mut result = Vec::with_capacity(ds.n_obs);
    for i in 0..ds.n_obs {
        let mut min = f64::INFINITY;
        let mut found = false;
        for name in vars {
            if let Some(v) = ds.get_var(name) {
                if let Some(val) = v.data[i].as_f64() {
                    if !val.is_nan() {
                        min = min.min(val);
                        found = true;
                    }
                }
            }
        }
        result.push(Value::Double(if found { min } else { f64::NAN }));
    }
    Ok(result)
}

fn egen_rowmiss(ds: &DataSet, vars: &[String]) -> Result<Vec<Value>, String> {
    let mut result = Vec::with_capacity(ds.n_obs);
    for i in 0..ds.n_obs {
        let mut miss = 0;
        for name in vars {
            if let Some(v) = ds.get_var(name) {
                if v.data[i].is_missing() {
                    miss += 1;
                }
            }
        }
        result.push(Value::Double(miss as f64));
    }
    Ok(result)
}

fn egen_rownonmiss(ds: &DataSet, vars: &[String]) -> Result<Vec<Value>, String> {
    let mut result = Vec::with_capacity(ds.n_obs);
    for i in 0..ds.n_obs {
        let mut nonmiss = 0;
        for name in vars {
            if let Some(v) = ds.get_var(name) {
                if !v.data[i].is_missing() {
                    nonmiss += 1;
                }
            }
        }
        result.push(Value::Double(nonmiss as f64));
    }
    Ok(result)
}

fn egen_group(ds: &DataSet, vars: &[String]) -> Result<Vec<Value>, String> {
    let mut groups: HashMap<Vec<String>, i32> = HashMap::new();
    let mut next_group = 1i32;
    let mut result = Vec::with_capacity(ds.n_obs);

    for i in 0..ds.n_obs {
        let mut key = Vec::new();
        for name in vars {
            if let Some(v) = ds.get_var(name) {
                let s = match &v.data[i] {
                    Value::Str(s) => s.clone(),
                    Value::Double(v) => format!("{:.10}", v),
                    Value::Float(v) => format!("{:.10}", v),
                    Value::Int(v) => v.to_string(),
                    Value::Long(v) => v.to_string(),
                    Value::Byte(v) => v.to_string(),
                    Value::Missing => ".".to_string(),
                    _ => "".to_string(),
                };
                key.push(s);
            }
        }
        let group = *groups.entry(key).or_insert_with(|| {
            let g = next_group;
            next_group += 1;
            g
        });
        result.push(Value::Long(group));
    }
    Ok(result)
}

fn egen_seq(ds: &DataSet, _args: &[String]) -> Result<Vec<Value>, String> {
    let mut result = Vec::with_capacity(ds.n_obs);
    for i in 1..=ds.n_obs {
        result.push(Value::Long(i as i32));
    }
    Ok(result)
}

fn egen_rank(ds: &DataSet, args: &[String]) -> Result<Vec<Value>, String> {
    if args.is_empty() {
        return Err("rank requires a variable".into());
    }
    let var = ds.get_var(&args[0]).ok_or_else(|| format!("variable {} not found", &args[0]))?;
    let mut pairs: Vec<(f64, usize)> = Vec::new();
    for i in 0..ds.n_obs {
        if let Some(v) = var.data[i].as_f64() {
            if !v.is_nan() {
                pairs.push((v, i));
            }
        }
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut result = vec![Value::Double(f64::NAN); ds.n_obs];
    for (rank, (_, idx)) in pairs.iter().enumerate() {
        result[*idx] = Value::Double((rank + 1) as f64);
    }
    Ok(result)
}

fn egen_count(ds: &DataSet, args: &[String]) -> Result<Vec<Value>, String> {
    // Count non-missing for each group or overall
    let n = if args.is_empty() {
        ds.n_obs
    } else {
        let mut count = 0;
        for i in 0..ds.n_obs {
            if let Some(v) = ds.get_var(&args[0]) {
                if !v.data[i].is_missing() {
                    count += 1;
                }
            }
        }
        count
    };
    Ok(vec![Value::Double(n as f64); ds.n_obs])
}

fn egen_mean(ds: &DataSet, args: &[String]) -> Result<Vec<Value>, String> {
    if args.is_empty() {
        return Err("mean requires a variable".into());
    }
    let vals: Vec<f64> = ds.get_var(&args[0])
        .ok_or_else(|| format!("variable {} not found", &args[0]))?
        .as_f64_vec().unwrap_or_default()
        .into_iter().filter(|v| !v.is_nan()).collect();
    let mean = if vals.is_empty() { f64::NAN } else { vals.iter().sum::<f64>() / vals.len() as f64 };
    Ok(vec![Value::Double(mean); ds.n_obs])
}

fn egen_sd(ds: &DataSet, args: &[String]) -> Result<Vec<Value>, String> {
    if args.is_empty() {
        return Err("sd requires a variable".into());
    }
    let vals: Vec<f64> = ds.get_var(&args[0])
        .ok_or_else(|| format!("variable {} not found", &args[0]))?
        .as_f64_vec().unwrap_or_default()
        .into_iter().filter(|v| !v.is_nan()).collect();
    let n = vals.len() as f64;
    let sd = if n < 2.0 { f64::NAN } else {
        let mean = vals.iter().sum::<f64>() / n;
        (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    };
    Ok(vec![Value::Double(sd); ds.n_obs])
}

fn egen_max(ds: &DataSet, args: &[String]) -> Result<Vec<Value>, String> {
    if args.is_empty() {
        return Err("max requires a variable".into());
    }
    let max = ds.get_var(&args[0])
        .ok_or_else(|| format!("variable {} not found", &args[0]))?
        .as_f64_vec().unwrap_or_default()
        .into_iter().filter(|v| !v.is_nan())
        .fold(f64::NEG_INFINITY, f64::max);
    Ok(vec![Value::Double(max); ds.n_obs])
}

fn egen_min(ds: &DataSet, args: &[String]) -> Result<Vec<Value>, String> {
    if args.is_empty() {
        return Err("min requires a variable".into());
    }
    let min = ds.get_var(&args[0])
        .ok_or_else(|| format!("variable {} not found", &args[0]))?
        .as_f64_vec().unwrap_or_default()
        .into_iter().filter(|v| !v.is_nan())
        .fold(f64::INFINITY, f64::min);
    Ok(vec![Value::Double(min); ds.n_obs])
}

fn egen_total(ds: &DataSet, args: &[String]) -> Result<Vec<Value>, String> {
    if args.is_empty() {
        return Err("total requires a variable".into());
    }
    let total = ds.get_var(&args[0])
        .ok_or_else(|| format!("variable {} not found", &args[0]))?
        .as_f64_vec().unwrap_or_default()
        .into_iter().filter(|v| !v.is_nan())
        .sum::<f64>();
    Ok(vec![Value::Double(total); ds.n_obs])
}
