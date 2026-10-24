use ferrostat_core::DataSet;
use ferrostat_parser::Command;
use crate::commands::CommandHandler;

pub struct ListCmd;

impl CommandHandler for ListCmd {
    fn name(&self) -> &'static str { "list" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        let varlist: Vec<String> = if cmd.args.is_empty() {
            ds.varlist()
        } else {
            let mut vars = Vec::new();
            for arg in &cmd.args {
                if arg.eq_ignore_ascii_case("in") || arg.eq_ignore_ascii_case("if") {
                    break;
                }
                vars.push(arg.clone());
            }
            if vars.is_empty() { ds.varlist() } else { vars }
        };

        let mut rows: Vec<usize> = (0..ds.n_obs).collect();
        if let Some(ref range) = cmd.in_range {
            rows = super::summarize::apply_in(ds, range)?;
        }
        if let Some(ref expr) = cmd.if_expr {
            let mask_rows: std::collections::HashSet<usize> = super::summarize::apply_if(ds, expr)?.into_iter().collect();
            rows.retain(|r| mask_rows.contains(r));
        }

        let mut out = Vec::new();
        // Header
        out.push(varlist.join("  "));
        out.push("-".repeat(60));

        for &i in &rows {
            let mut line_parts = Vec::new();
            for name in &varlist {
                if let Some(var) = ds.get_var(name) {
                    let val_str = match &var.data[i] {
                        ferrostat_core::Value::Str(s) => s.clone(),
                        ferrostat_core::Value::Double(v) => {
                            if v.is_nan() { ".".to_string() } else { format!("{:.4}", v) }
                        }
                        ferrostat_core::Value::Float(v) => {
                            if v.is_nan() { ".".to_string() } else { format!("{:.4}", v) }
                        }
                        ferrostat_core::Value::Int(v) => v.to_string(),
                        ferrostat_core::Value::Long(v) => v.to_string(),
                        ferrostat_core::Value::Byte(v) => v.to_string(),
                        ferrostat_core::Value::StrL(v, o) => {
                            ds.resolve_strl(*v, *o).unwrap_or("(strL)").to_string()
                        }
                        ferrostat_core::Value::Missing => ".".to_string(),
                    };
                    line_parts.push(val_str);
                }
            }
            out.push(line_parts.join("  "));
        }

        Ok(out.join("\n"))
    }
}
