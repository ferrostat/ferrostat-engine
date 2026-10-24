use std::collections::{BTreeMap, HashMap};
use crate::var::{Variable, Value};
use crate::value_label::ValueLabel;

/// Core Ferrostat dataset: rectangular matrix of variables and observations.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DataSet {
    /// Variables in order of creation.
    pub vars: Vec<Variable>,
    /// Variable name -> index.
    pub var_index: HashMap<String, usize>,
    /// Number of observations.
    pub n_obs: usize,
    /// Current sort order (variable names).
    pub sort_by: Vec<String>,
    /// Dataset label / note.
    pub datalabel: String,
    /// Timestamp from file.
    pub time_stamp: String,
    /// Characteristics: [varname][charname] = value. "_dta" for dataset chars.
    pub characteristics: BTreeMap<String, BTreeMap<String, String>>,
    /// Value label definitions by name.
    pub value_labels: HashMap<String, ValueLabel>,
    /// strL table: (v, o) -> string content.
    pub strls: HashMap<(u32, u32), String>,
}

impl DataSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a variable. If dataset already has observations, pad with missing.
    pub fn add_variable(&mut self, mut var: Variable) -> Result<(), String> {
        if self.var_index.contains_key(&var.name) {
            return Err(format!("variable {} already defined", var.name));
        }
        while var.len() < self.n_obs {
            var.push_missing();
        }
        self.var_index.insert(var.name.clone(), self.vars.len());
        self.vars.push(var);
        Ok(())
    }

    /// Drop a variable by name.
    pub fn drop_variable(&mut self, name: &str) -> Result<(), String> {
        let idx = self.var_index.remove(name)
            .ok_or_else(|| format!("variable {} not found", name))?;
        self.vars.remove(idx);
        // Rebuild index
        self.var_index.clear();
        for (i, v) in self.vars.iter().enumerate() {
            self.var_index.insert(v.name.clone(), i);
        }
        self.sort_by.retain(|n| n != name);
        Ok(())
    }

    /// Keep only specified variables.
    pub fn keep_variables(&mut self, names: &[String]) {
        let to_drop: Vec<String> = self.vars.iter()
            .map(|v| v.name.clone())
            .filter(|n| !names.contains(n))
            .collect();
        for n in to_drop {
            let _ = self.drop_variable(&n);
        }
    }

    /// Access a variable by name.
    pub fn get_var(&self, name: &str) -> Option<&Variable> {
        self.var_index.get(name).map(|&i| &self.vars[i])
    }

    pub fn get_var_mut(&mut self, name: &str) -> Option<&mut Variable> {
        let idx = *self.var_index.get(name)?;
        Some(&mut self.vars[idx])
    }

    /// Number of variables.
    pub fn n_vars(&self) -> usize {
        self.vars.len()
    }

    /// Add an empty observation.
    pub fn add_observation(&mut self) {
        for var in &mut self.vars {
            var.push_missing();
        }
        self.n_obs += 1;
    }

    /// Drop an observation by index.
    pub fn drop_observation(&mut self, idx: usize) -> Result<(), String> {
        if idx >= self.n_obs {
            return Err(format!("observation index {} out of range", idx));
        }
        for var in &mut self.vars {
            var.remove(idx);
        }
        self.n_obs -= 1;
        Ok(())
    }

    /// Keep observations where mask is true.
    pub fn keep_if(&mut self, mask: &[bool]) -> Result<(), String> {
        if mask.len() != self.n_obs {
            return Err(format!("mask length {} does not match n_obs {}", mask.len(), self.n_obs));
        }
        for var in &mut self.vars {
            let new_data: Vec<Value> = std::mem::take(&mut var.data)
                .into_iter()
                .zip(mask.iter())
                .filter(|&(_, m)| *m)
                .map(|(v, _)| v)
                .collect();
            var.data = new_data;
        }
        self.n_obs = mask.iter().filter(|&&m| m).count();
        Ok(())
    }

    /// Sort dataset by variable list (ascending, missing last).
    pub fn sort(&mut self, varlist: &[String], stable: bool) -> Result<(), String> {
        if varlist.is_empty() {
            return Ok(());
        }
        let mut cmp_indices: Vec<usize> = (0..self.n_obs).collect();
        if stable {
            cmp_indices.sort_by(|&a, &b| self.cmp_obs(a, b, varlist));
        } else {
            cmp_indices.sort_unstable_by(|&a, &b| self.cmp_obs(a, b, varlist));
        }
        // Reorder each variable
        for var in &mut self.vars {
            let new_data: Vec<Value> = cmp_indices.iter()
                .map(|&i| var.data[i].clone())
                .collect();
            var.data = new_data;
        }
        self.sort_by = varlist.to_vec();
        Ok(())
    }

    fn cmp_obs(&self, a: usize, b: usize, varlist: &[String]) -> std::cmp::Ordering {
        for name in varlist {
            let var = self.get_var(name).expect("sort var exists");
            let va = &var.data[a];
            let vb = &var.data[b];
            let ord = va.partial_cmp(vb).unwrap_or(std::cmp::Ordering::Equal);
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
        }
        std::cmp::Ordering::Equal
    }

    /// Get observation as a map.
    pub fn get_observation(&self, idx: usize) -> Option<HashMap<String, Value>> {
        if idx >= self.n_obs {
            return None;
        }
        let mut row = HashMap::new();
        for var in &self.vars {
            row.insert(var.name.clone(), var.data[idx].clone());
        }
        Some(row)
    }

    /// Get a column by name.
    pub fn get_column(&self, name: &str) -> Option<Vec<Value>> {
        self.get_var(name).map(|v| v.data.clone())
    }

    /// Set a column by name.
    pub fn set_column(&mut self, name: &str, values: Vec<Value>) -> Result<(), String> {
        let idx = *self.var_index.get(name)
            .ok_or_else(|| format!("variable {} not found", name))?;
        if values.len() != self.n_obs {
            return Err(format!("Length mismatch: {} vs {}", values.len(), self.n_obs));
        }
        self.vars[idx].data = values;
        Ok(())
    }

    /// Return list of variable names in order.
    pub fn varlist(&self) -> Vec<String> {
        self.vars.iter().map(|v| v.name.clone()).collect()
    }

    /// Format a describe output.
    pub fn describe(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Contains data".to_string());
        lines.push(format!("  obs: {}", self.n_obs));
        lines.push(format!(" vars: {}", self.n_vars()));
        lines.push("".to_string());
        lines.push("              storage   display    value".to_string());
        lines.push("variable name   type    format     label      variable label".to_string());
        lines.push("-".repeat(60));
        for var in &self.vars {
            let vlab = if var.value_label_name.is_empty() {
                "".to_string()
            } else {
                var.value_label_name.clone()
            };
            lines.push(format!(
                "{:15} {:7} {:10} {:10} {}",
                var.name, var.var_type.type_str(), var.format, vlab, var.label
            ));
        }
        lines.join("\n")
    }

    pub fn resolve_strl(&self, v: u32, o: u32) -> Option<&str> {
        self.strls.get(&(v, o)).map(|s| s.as_str())
    }
}
