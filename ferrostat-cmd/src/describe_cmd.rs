use ferrostat_core::DataSet;
use ferrostat_parser::parser::Command;
use crate::commands::CommandHandler;

pub struct DescribeCmd;

impl CommandHandler for DescribeCmd {
    fn name(&self) -> &'static str { "describe" }

    fn execute(&self, ds: &mut DataSet, _cmd: &Command) -> Result<String, String> {
        Ok(ds.describe())
    }
}
