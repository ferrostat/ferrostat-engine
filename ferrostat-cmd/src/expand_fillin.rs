use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::Command;
use crate::commands::CommandHandler;

/// Expand: duplicate observations.
pub struct Expand;

impl CommandHandler for Expand {
    fn name(&self) -> &'static str { "expand" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.is_empty() {
            return Err("expand requires an expression".into());
        }

        let expr = cmd.args.join(" ");
        // Simple case: expand N (constant)
        if let Ok(n) = expr.parse::<usize>() {
            let old_obs = ds.n_obs;
            let mut new_vars = Vec::new();
            for var in &ds.vars {
                let mut nv = Variable::new(var.name.clone(), var.var_type);
                nv.format = var.format.clone();
                nv.label = var.label.clone();
                nv.value_label_name = var.value_label_name.clone();
                for _ in 0..n {
                    for val in &var.data {
                        nv.data.push(val.clone());
                    }
                }
                new_vars.push(nv);
            }
            ds.vars = new_vars;
            ds.var_index.clear();
            for (i, v) in ds.vars.iter().enumerate() {
                ds.var_index.insert(v.name.clone(), i);
            }
            ds.n_obs = old_obs * n;
            return Ok(format!("expanded to {} observations", ds.n_obs));
        }

        // Expand by variable
        if let Some(var) = ds.get_var(&expr) {
            let mut new_vars: Vec<Variable> = ds.vars.iter().map(|v| {
                let mut nv = Variable::new(v.name.clone(), v.var_type);
                nv.format = v.format.clone();
                nv.label = v.label.clone();
                nv.value_label_name = v.value_label_name.clone();
                nv
            }).collect();

            for i in 0..ds.n_obs {
                let n = match &var.data[i] {
                    Value::Double(v) => v.round() as usize,
                    Value::Float(v) => v.round() as usize,
                    Value::Int(v) => *v as usize,
                    Value::Long(v) => *v as usize,
                    Value::Byte(v) => *v as usize,
                    _ => 1,
                };
                for _ in 0..n {
                    for (j, old_var) in ds.vars.iter().enumerate() {
                        new_vars[j].data.push(old_var.data[i].clone());
                    }
                }
            }

            ds.vars = new_vars;
            ds.var_index.clear();
            for (i, v) in ds.vars.iter().enumerate() {
                ds.var_index.insert(v.name.clone(), i);
            }
            ds.n_obs = ds.vars[0].data.len();
            Ok(format!("expanded to {} observations", ds.n_obs))
        } else {
            Err(format!("variable {} not found", expr))
        }
    }
}

/// Fillin: creates all combinations of specified variables.
pub struct Fillin;

impl CommandHandler for Fillin {
    fn name(&self) -> &'static str { "fillin" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.is_empty() {
            return Err("fillin requires a varlist".into());
        }

        let mut all_levels: Vec<Vec<String>> = Vec::new();
        for name in &cmd.args {
            let var = ds.get_var(name).ok_or_else(|| format!("variable {} not found", name))?;
            let mut levels: Vec<String> = Vec::new();
            for val in &var.data {
                let s = match val {
                    Value::Str(s) => s.clone(),
                    Value::Double(v) => format!("{}", v),
                    Value::Float(v) => format!("{}", v),
                    Value::Int(v) => v.to_string(),
                    Value::Long(v) => v.to_string(),
                    Value::Byte(v) => v.to_string(),
                    _ => "".to_string(),
                };
                if !levels.contains(&s) {
                    levels.push(s);
                }
            }
            levels.sort();
            all_levels.push(levels);
        }

        // Generate all combinations
        let mut combinations: Vec<Vec<String>> = vec![vec![]];
        for levels in &all_levels {
            let mut new_comb = Vec::new();
            for comb in &combinations {
                for level in levels {
                    let mut new = comb.clone();
                    new.push(level.clone());
                    new_comb.push(new);
                }
            }
            combinations = new_comb;
        }

        // Build new dataset with _fillin marker
        let mut new_ds = DataSet::new();
        for name in &cmd.args {
            let old = ds.get_var(name).unwrap();
            let mut nv = Variable::new(name.clone(), old.var_type);
            new_ds.add_variable(nv).unwrap();
        }

        // Add other variables
        for var in &ds.vars {
            if !cmd.args.contains(&var.name) {
                let mut nv = Variable::new(var.name.clone(), var.var_type);
                new_ds.add_variable(nv).unwrap();
            }
        }

        // Add _fillin
        let fillin_var = Variable::new("_fillin", VarType::Byte);
        new_ds.add_variable(fillin_var).unwrap();

        // Find which combinations already exist
        let mut existing: std::collections::HashSet<Vec<String>> = std::collections::HashSet::new();
        for i in 0..ds.n_obs {
            let mut key = Vec::new();
            for name in &cmd.args {
                let val_str = match &ds.get_var(name).unwrap().data[i] {
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
            existing.insert(key);
        }

        // Populate
        for comb in &combinations {
            let is_new = !existing.contains(comb);
            for (j, name) in cmd.args.iter().enumerate() {
                let old = ds.get_var(name).unwrap();
                let val = match old.var_type {
                    VarType::Str(_) | VarType::StrL => Value::Str(comb[j].clone()),
                    _ => {
                        if let Ok(v) = comb[j].parse::<f64>() {
                            Value::Double(v)
                        } else {
                            Value::Missing
                        }
                    }
                };
                new_ds.vars[j].data.push(val);
            }
            // Other variables
            for var in &ds.vars {
                if !cmd.args.contains(&var.name) {
                    let idx = new_ds.var_index[&var.name];
                    if is_new {
                        new_ds.vars[idx].data.push(Value::Missing);
                    } else {
                        new_ds.vars[idx].data.push(var.data[0].clone());
                    }
                }
            }
            let fill_idx = new_ds.var_index["_fillin"];
            new_ds.vars[fill_idx].data.push(Value::Byte(if is_new { 1 } else { 0 }));
        }

        new_ds.n_obs = combinations.len();
        *ds = new_ds;
        Ok(format!("filled in {} combinations", combinations.len()))
    }
}
