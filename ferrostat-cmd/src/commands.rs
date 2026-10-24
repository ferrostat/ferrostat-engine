use ferrostat_core::DataSet;
use ferrostat_parser::parser::Command;

pub trait CommandHandler {
    fn name(&self) -> &'static str;
    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String>;
}
