use std::collections::HashMap;

use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::Command;
use crate::commands::CommandHandler;

/// Contract data to a frequency table.
pub struct Contract;

impl CommandHandler for Contract {
    fn name(&self) -> &'static str { "contract" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.is_empty() {
            return Err("contract requires a varlist".into());
        }

        let varlist = &cmd.args;
        for name in varlist {
            if ds.get_var(name).is_none() {
                return Err(format!("variable {} not found", name));
            }
        }

        // Build frequency table
        let mut freq_map: HashMap<Vec<String>, usize> = HashMap::new();
        for i in 0..ds.n_obs {
            let mut key = Vec::new();
            for name in varlist {
                let val = match &ds.get_var(name).unwrap().data[i] {
                    Value::Str(s) => s.clone(),
                    Value::Double(v) => format!("{:.10}", v),
                    Value::Float(v) => format!("{:.10}", v),
                    Value::Int(v) => v.to_string(),
                    Value::Long(v) => v.to_string(),
                    Value::Byte(v) => v.to_string(),
                    Value::StrL(v, o) => ds.resolve_strl(*v, *o).unwrap_or("(strL)").to_string(),
                    Value::Missing => ".".to_string(),
                };
                key.push(val);
            }
            *freq_map.entry(key).or_insert(0) += 1;
        }

        // Build new dataset
        let mut new_ds = DataSet::new();
        for name in varlist {
            let old_var = ds.get_var(name).unwrap();
            let mut new_var = Variable::new(name.clone(), old_var.var_type);
            new_var.format = old_var.format.clone();
            new_var.label = old_var.label.clone();
            new_var.value_label_name = old_var.value_label_name.clone();
            new_ds.add_variable(new_var).unwrap();
        }

        // Add _freq variable
        let mut freq_var = Variable::new("_freq", VarType::Long);
        freq_var.format = "%8.0g".to_string();
        new_ds.add_variable(freq_var).unwrap();

        for (key, count) in &freq_map {
            for (j, name) in varlist.iter().enumerate() {
                let old_var = ds.get_var(name).unwrap();
                let val = match old_var.var_type {
                    VarType::Str(_) | VarType::StrL => Value::Str(key[j].clone()),
                    _ => {
                        if let Ok(v) = key[j].parse::<f64>() {
                            Value::Double(v)
                        } else {
                            Value::Missing
                        }
                    }
                };
                new_ds.vars[j].data.push(val);
            }
            new_ds.vars[varlist.len()].data.push(Value::Long(*count as i32));
        }

        new_ds.n_obs = new_ds.vars[0].data.len();
        *ds = new_ds;
        Ok(format!("contracted to {} unique combinations", freq_map.len()))
    }
}
