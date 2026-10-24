use ferrostat_core::DataSet;
use ferrostat_parser::parser::Command;

use crate::store::Store;

/// A command: receives the data in memory, the interpreter state and the
/// parsed command line; returns the text to display.
pub trait CommandHandler {
    fn name(&self) -> &'static str;
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String>;
}
