use ferrostat_core::DataSet;
use ferrostat_parser::Command;
use crate::commands::CommandHandler;

/// Display text or expression results.
pub struct DisplayCmd;

impl CommandHandler for DisplayCmd {
    fn name(&self) -> &'static str { "display" }

    fn execute(&self, ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        let text = cmd.args.join(" ");
        // Simple macro replacement
        let mut text = text;
        text = text.replace("\"", "");
        Ok(text)
    }
}

/// Local and global macros.
pub struct LocalMacro;

impl CommandHandler for LocalMacro {
    fn name(&self) -> &'static str { "local" }

    fn execute(&self, _ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.len() < 2 {
            return Err("local requires: local macname = expression".into());
        }
        let macname = &cmd.args[0];
        if cmd.args[1] != "=" {
            return Err("local syntax: local macname = expression".into());
        }
        let value = cmd.args[2..].join(" ");
        // We can't actually store since CommandHandler doesn't have access to macro_vars
        // The engine handles this directly for now
        Ok(format!("local {} = {}", macname, value))
    }
}

/// By-prefix processing.
pub struct ByPrefix;

impl CommandHandler for ByPrefix {
    fn name(&self) -> &'static str { "by" }

    fn execute(&self, _ds: &mut DataSet, cmd: &Command) -> Result<String, String> {
        if cmd.args.is_empty() {
            return Err("by requires varlist: command".into());
        }
        Ok("by prefix not yet implemented in full; use direct commands".into())
    }
}
