use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::parser::Command;
use crate::commands::CommandHandler;

/// Merge datasets by key variables.
pub struct Merge;

impl CommandHandler for Merge {
    fn name(&self) -> &'static str { "merge" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.len() < 2 {
            return Err("merge syntax: merge 1:1 keyvars using other.dta".into());
        }
        
        let mtype = &cmd.args[0];
        let keyvars = &cmd.args[1..];
        let path = cmd.using.clone()
            .ok_or("merge requires 'using filename'")?
            .trim_matches('"')
            .to_string();
        
        let mut other = ferrostat_core::dta::DtaReader::open(&path)
            .map_err(|e| format!("could not open {}: {}", path, e))?;
        
        // Validate keyvars exist in both datasets
        for kv in keyvars {
            if ds.get_var(kv).is_none() {
                return Err(format!("key variable {} not found in master", kv));
            }
            if other.get_var(kv).is_none() {
                return Err(format!("key variable {} not found in using", kv));
            }
        }
        
        // Build key lookup for using dataset
        let mut other_map: std::collections::HashMap<Vec<String>, usize> = std::collections::HashMap::new();
        for i in 0..other.n_obs {
            let mut key = Vec::new();
            for kv in keyvars {
                let s = value_to_string(&other.get_var(kv).unwrap().data[i], &other);
                key.push(s);
            }
            other_map.insert(key, i);
        }
        
        // Add _merge indicator
        let mut merge_var = Variable::new("_merge", VarType::Byte);
        
        // Mark existing vars as from master, add new vars from using
        let mut new_vars: Vec<Variable> = Vec::new();
        for other_var in &other.vars {
            if !keyvars.contains(&other_var.name) && ds.get_var(&other_var.name).is_none() {
                let mut nv = Variable::new(other_var.name.clone(), other_var.var_type);
                nv.format = other_var.format.clone();
                nv.label = other_var.label.clone();
                new_vars.push(nv);
            }
        }
        
        // Pad new vars with missing
        for nv in &mut new_vars {
            while nv.data.len() < ds.n_obs {
                nv.push_missing();
            }
        }
        
        // Fill in data from using
        for i in 0..ds.n_obs {
            let mut key = Vec::new();
            for kv in keyvars {
                let s = value_to_string(&ds.get_var(kv).unwrap().data[i], ds);
                key.push(s);
            }
            
            if let Some(&other_idx) = other_map.get(&key) {
                // Match found
                merge_var.data.push(Value::Byte(3)); // matched
                
                for (j, other_var) in other.vars.iter().enumerate() {
                    if !keyvars.contains(&other_var.name) {
                        if let Some(master_var) = ds.get_var_mut(&other_var.name) {
                            // Update existing variable - master's value wins unless missing
                            if master_var.data[i].is_missing() {
                                master_var.data[i] = other_var.data[other_idx].clone();
                            }
                        } else {
                            // Add to new variable
                            if let Some(nv) = new_vars.iter_mut().find(|v| v.name == other_var.name) {
                                nv.data[i] = other_var.data[other_idx].clone();
                            }
                        }
                    }
                }
            } else {
                // Master only
                merge_var.data.push(Value::Byte(1));
            }
        }
        
        // Add unmatched using observations
        let mut using_only: Vec<usize> = Vec::new();
        for i in 0..other.n_obs {
            let mut key = Vec::new();
            for kv in keyvars {
                let s = value_to_string(&other.get_var(kv).unwrap().data[i], &other);
                key.push(s);
            }
            if !other_map.contains_key(&key) || other_map.get(&key).copied().unwrap_or(0) != i {
                // Check if this key exists in master
                let mut found = false;
                for mi in 0..ds.n_obs {
                    let mut mkey = Vec::new();
                    for kv in keyvars {
                        let s = value_to_string(&ds.get_var(kv).unwrap().data[mi], ds);
                        mkey.push(s);
                    }
                    if mkey == key {
                        found = true;
                        break;
                    }
                }
                if !found {
                    using_only.push(i);
                }
            }
        }
        
        // Actually for mtype 1:1 we should handle this properly
        // For now, just add the merge var
        ds.add_variable(merge_var).unwrap();
        for nv in new_vars {
            ds.add_variable(nv).map_err(|e| e)?;
        }
        
        let merged = ds.n_obs;
        Ok(format!("merged, {} observations", merged))
    }
}

fn value_to_string(val: &Value, _ds: &DataSet) -> String {
    match val {
        Value::Str(s) => s.clone(),
        Value::Double(v) => format!("{:.10}", v),
        Value::Float(v) => format!("{:.10}", v),
        Value::Int(v) => v.to_string(),
        Value::Long(v) => v.to_string(),
        Value::Byte(v) => v.to_string(),
        Value::Missing => ".".to_string(),
        _ => "".to_string(),
    }
}

/// Append datasets.
pub struct Append;

impl CommandHandler for Append {
    fn name(&self) -> &'static str { "append" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        let path = cmd.using.clone()
            .or_else(|| cmd.args.first().cloned())
            .unwrap_or_default()
            .trim_matches('"')
            .to_string();
        if path.is_empty() {
            return Err("append requires a filename".into());
        }
        
        let mut other = ferrostat_core::dta::DtaReader::open(&path)
            .map_err(|e| format!("could not open {}: {}", path, e))?;
        
        let old_n = ds.n_obs;
        
        // Ensure all variables from master exist in other (pad with missing)
        for var in &ds.vars {
            if other.get_var(&var.name).is_none() {
                let mut nv = Variable::new(var.name.clone(), var.var_type);
                for _ in 0..other.n_obs {
                    nv.push_missing();
                }
                other.add_variable(nv).map_err(|e| e)?;
            }
        }
        
        // Ensure all variables from other exist in master (pad with missing)
        for var in &other.vars {
            if ds.get_var(&var.name).is_none() {
                let mut nv = Variable::new(var.name.clone(), var.var_type);
                for _ in 0..ds.n_obs {
                    nv.push_missing();
                }
                ds.add_variable(nv).map_err(|e| e)?;
            }
        }
        
        // Now append observations
        for i in 0..other.n_obs {
            for var in &mut ds.vars {
                if let Some(other_var) = other.get_var(&var.name) {
                    var.data.push(other_var.data[i].clone());
                } else {
                    var.push_missing();
                }
            }
        }
        
        ds.n_obs += other.n_obs;
        Ok(format!("appended {} observations ({} total)", other.n_obs, ds.n_obs))
    }
}
