use ferrostat_core::DataSet;
use ferrostat_parser::Command;
use crate::commands::CommandHandler;

pub struct Count;

impl CommandHandler for Count {
    fn name(&self) -> &'static str { "count" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        let mut rows: Vec<usize> = (0..ds.n_obs).collect();
        if let Some(ref expr) = cmd.if_expr {
            rows = crate::summarize::apply_if(ds, expr)?;
        }
        if let Some(ref range) = cmd.in_range {
            rows = crate::summarize::apply_in(ds, range)?;
        }
        Ok(format!("  {}\n", rows.len()))
    }
}
