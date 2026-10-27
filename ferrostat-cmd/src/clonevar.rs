//! `clonevar` (exact copy of a variable with all its metadata) and the
//! interactive data viewers `browse`/`edit`, which are no-ops in batch mode.

use ferrostat_core::{DataSet, Value, Variable};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::store::Store;
use crate::{util, varlist};

pub struct Clonevar;

impl CommandHandler for Clonevar {
    fn name(&self) -> &'static str {
        "clonevar"
    }

    /// `clonevar newvar = oldvar [if] [in]`: same type, display format,
    /// variable label, value label and characteristics; observations outside
    /// the selected sample are missing.
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let main = cmd.main.trim();
        let (lhs, rhs) = main.split_once('=').ok_or_else(|| rc_err(198, "= required"))?;
        let rhs = rhs.trim().trim_start_matches('=').trim();
        let new = match lhs.split_whitespace().collect::<Vec<_>>().as_slice() {
            [n] => n.to_string(),
            _ => return Err(rc_err(198, "invalid syntax")),
        };
        if rhs.split_whitespace().count() != 1 {
            return Err(rc_err(198, "invalid syntax"));
        }
        varlist::check_name(&new)?;
        if ds.var_index.contains_key(&new) {
            return Err(rc_err(110, format!("variable {new} already defined")));
        }
        let old = varlist::resolve_name(ds, rhs)?;
        let mask = util::touse(ds, st, cmd)?;
        let src = ds.get_var(&old).unwrap();
        let mut var = Variable::new(new.clone(), src.var_type);
        var.format = src.format.clone();
        var.label = src.label.clone();
        var.value_label_name = src.value_label_name.clone();
        let is_str = src.var_type.is_string();
        var.data = (0..ds.n_obs)
            .map(|i| match src.data.get(i) {
                // strL cells are resolved so the copy does not share references
                Some(Value::StrL(v, o)) if mask[i] => Value::Str(ds.resolve_strl(*v, *o).unwrap_or("").to_string()),
                Some(x) if mask[i] => x.clone(),
                _ if is_str => Value::Str(String::new()),
                _ => Value::Double(ferrostat_core::missing::SYSMISS),
            })
            .collect();
        // like `generate`, report every missing value of the new variable
        let n_miss = if is_str { 0 } else { var.data.iter().filter(|v| v.is_missing() || ferrostat_core::missing::is_missing(v.num())).count() };
        let chars = ds.characteristics.get(&old).cloned();
        ds.add_variable(var).map_err(|e| rc_err(110, e))?;
        if let Some(c) = chars {
            ds.characteristics.insert(new, c);
        }
        Ok(if n_miss > 0 { format!("({n_miss} missing value{} generated)", if n_miss == 1 { "" } else { "s" }) } else { String::new() })
    }
}

pub struct Browse;

impl CommandHandler for Browse {
    fn name(&self) -> &'static str {
        "browse"
    }

    /// `browse`/`edit [varlist] [if] [in]`: opens the data viewer in
    /// interactive front ends; in batch mode it only validates its arguments.
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        if !cmd.main.trim().is_empty() {
            varlist::expand(ds, &cmd.main)?;
        }
        util::touse(ds, st, cmd)?;
        Ok(String::new())
    }
}
