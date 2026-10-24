use std::collections::HashMap;

use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::Command;
use crate::commands::CommandHandler;

/// Reshape: convert between wide and long formats.
pub struct Reshape;

impl CommandHandler for Reshape {
    fn name(&self) -> &'static str { "reshape" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.is_empty() {
            return Err("reshape requires wide/long".into());
        }

        let direction = cmd.args[0].to_lowercase();
        match direction.as_str() {
            "long" => reshape_long(ds, cmd),
            "wide" => reshape_wide(ds, cmd),
            _ => Err("reshape direction must be wide or long".into()),
        }
    }
}

fn reshape_long(ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
    // Extract j variable and stub names
    // Syntax: reshape long stubnames, i(idvar) j(jvar)
    if cmd.args.len() < 2 {
        return Err("reshape long requires stubnames".into());
    }

    let stub = &cmd.args[1];
    let id_var = cmd.options.iter()
        .find(|(k, _)| k == "i")
        .and_then(|(_, v)| v.as_ref().and_then(crate::eval::expr_to_string))
        .unwrap_or_else(|| "id".to_string());

    // Find all variables matching stub followed by a number
    let mut j_values: Vec<String> = Vec::new();
    let mut matched_vars: Vec<(String, String)> = Vec::new(); // (varname, jvalue)

    for var in &ds.vars {
        let name = &var.name;
        if name.starts_with(stub) {
            let suffix = &name[stub.len()..];
            if !suffix.is_empty() {
                j_values.push(suffix.to_string());
                matched_vars.push((name.clone(), suffix.to_string()));
            }
        }
    }

    if matched_vars.is_empty() {
        return Err(format!("no variables found matching stub {}", stub));
    }

    j_values.sort();
    j_values.dedup();

    // Build new long dataset
    let mut new_ds = DataSet::new();

    // Add id variables
    for var in &ds.vars {
        if !var.name.starts_with(stub) || var.name == *stub {
            let nv = Variable::new(var.name.clone(), var.var_type);
            new_ds.add_variable(nv).unwrap();
        }
    }

    // Add j variable
    let j_var = Variable::new("j", VarType::Str(32));
    new_ds.add_variable(j_var).unwrap();

    // Add value variable
    let val_var = Variable::new(stub.clone(), VarType::Double);
    new_ds.add_variable(val_var).unwrap();

    for i in 0..ds.n_obs {
        for (var_name, j_val) in &matched_vars {
            // Copy id variables
            for var in &ds.vars {
                if !var.name.starts_with(stub) || var.name == *stub {
                    let idx = new_ds.var_index[&var.name];
                    new_ds.vars[idx].data.push(var.data[i].clone());
                }
            }
            // j
            let j_idx = new_ds.var_index["j"];
            new_ds.vars[j_idx].data.push(Value::Str(j_val.clone()));
            // value
            let v_idx = new_ds.var_index[stub];
            let val = ds.get_var(var_name).unwrap().data[i].clone();
            new_ds.vars[v_idx].data.push(val);
        }
    }

    new_ds.n_obs = ds.n_obs * matched_vars.len();
    *ds = new_ds;
    Ok(format!("data reshaped to long, {} observations", ds.n_obs))
}

fn reshape_wide(ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
    if cmd.args.len() < 2 {
        return Err("reshape wide requires stubnames".into());
    }

    let stub = &cmd.args[1];
    let j_var = cmd.options.iter()
        .find(|(k, _)| k == "j")
        .and_then(|(_, v)| v.as_ref().and_then(crate::eval::expr_to_string))
        .unwrap_or_else(|| "j".to_string());

    let id_var = cmd.options.iter()
        .find(|(k, _)| k == "i")
        .and_then(|(_, v)| v.as_ref().and_then(crate::eval::expr_to_string))
        .unwrap_or_else(|| "id".to_string());

    // Collect j values
    let mut j_values: Vec<String> = Vec::new();
    if let Some(jv) = ds.get_var(&j_var) {
        for val in &jv.data {
            let s = match val {
                Value::Str(s) => s.clone(),
                Value::Int(v) => v.to_string(),
                Value::Long(v) => v.to_string(),
                _ => continue,
            };
            if !j_values.contains(&s) {
                j_values.push(s);
            }
        }
    }
    j_values.sort();

    // Group by id
    let mut groups: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
    if let Some(_idv) = ds.get_var(&id_var) {
        for i in 0..ds.n_obs {
            let mut key = Vec::new();
            for name in id_var.split_whitespace() {
                if let Some(v) = ds.get_var(name) {
                    let s = match &v.data[i] {
                        Value::Str(s) => s.clone(),
                        Value::Int(v) => v.to_string(),
                        Value::Long(v) => v.to_string(),
                        Value::Double(v) => format!("{:.10}", v),
                        _ => "".to_string(),
                    };
                    key.push(s);
                }
            }
            groups.entry(key).or_insert_with(Vec::new).push(i);
        }
    }

    // Build wide dataset
    let mut new_ds = DataSet::new();

    // Add id variables
    for name in id_var.split_whitespace() {
        if let Some(v) = ds.get_var(name) {
            let nv = Variable::new(name.to_string(), v.var_type);
            new_ds.add_variable(nv).unwrap();
        }
    }

    // Add stub_j variables
    for j in &j_values {
        let nv = Variable::new(format!("{}{}", stub, j), VarType::Double);
        new_ds.add_variable(nv).unwrap();
    }

    // Populate
    for (_key, indices) in &groups {
        // Id values
        for (k, name) in id_var.split_whitespace().enumerate() {
            let idx = new_ds.var_index[name];
            if let Some(v) = ds.get_var(name) {
                new_ds.vars[idx].data.push(v.data[indices[0]].clone());
            }
        }
        // Stub values
        for j in &j_values {
            let var_name = format!("{}{}", stub, j);
            let idx = new_ds.var_index[&var_name];
            let mut found = false;
            for &i in indices {
                if let Some(jv) = ds.get_var(&j_var) {
                    let jval = match &jv.data[i] {
                        Value::Str(s) => s.clone(),
                        Value::Int(v) => v.to_string(),
                        _ => continue,
                    };
                    if &jval == j {
                        if let Some(sv) = ds.get_var(stub) {
                            new_ds.vars[idx].data.push(sv.data[i].clone());
                            found = true;
                            break;
                        }
                    }
                }
            }
            if !found {
                new_ds.vars[idx].data.push(Value::Missing);
            }
        }
    }

    if new_ds.vars.is_empty() {
        return Err("reshape resulted in empty dataset: no matching variable names found".into());
    }
    new_ds.n_obs = new_ds.vars[0].data.len();
    *ds = new_ds;
    Ok(format!("data reshaped to wide, {} observations", ds.n_obs))
}
