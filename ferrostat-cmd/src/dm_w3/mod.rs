//! Remaining data-management commands: diagnosis/procedure codes (`icd9`,
//! `icd10`, ...), variable lists (`vl`), business calendars (`bcal`),
//! `memory`, `snapshot`, `sysuse`, frame sets on disk and alias variables,
//! `import fred`, `export dbase`, and clear errors for database connectors
//! that are not available (`odbc`, `jdbc`, `import haver`).

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::rc_err;
use crate::store::Store;

pub mod bcal;
pub mod frames_more;
pub mod fred;
pub mod icd;
pub mod memsnap;
pub mod sysdata;
pub mod vl;

/// Database connectors that need external drivers.
struct Unsupported(&'static str, &'static str);

impl CommandHandler for Unsupported {
    fn name(&self) -> &'static str {
        self.0
    }
    fn execute(&self, _ds: &mut DataSet, _st: &mut Store, _cmd: &Command) -> Result<String, String> {
        Err(rc_err(199, format!("{}: {} is not supported in this build", self.0, self.1)))
    }
}

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    let mut v: Vec<(&'static str, Box<dyn CommandHandler>)> = icd::handlers();
    v.push(("vl", Box::new(vl::Vl)));
    v.push(("bcal", Box::new(bcal::Bcal)));
    v.push(("memory", Box::new(memsnap::Memory)));
    v.push(("snapshot", Box::new(memsnap::SnapshotCmd)));
    v.push(("fralias", Box::new(frames_more::Fralias)));
    v.push(("frunalias", Box::new(frames_more::Frunalias)));
    v.push(("odbc", Box::new(Unsupported("odbc", "access to ODBC data sources (no ODBC driver manager is linked)"))));
    v.push(("jdbc", Box::new(Unsupported("jdbc", "access to JDBC data sources (no Java runtime is linked)"))));
    v
}

/// `import <kind>` formats handled here.
pub fn import_cmd(kind: &str, rest: &str, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Option<Result<String, String>> {
    match kind {
        "fred" => Some(fred::import(rest, ds, st, cmd)),
        "haver" => Some(Err(rc_err(199, "import haver: Haver Analytics databases are not supported in this build"))),
        _ => None,
    }
}

/// `export <kind>` formats handled here.
pub fn export_cmd(kind: &str, rest: &str, ds: &DataSet, st: &Store, cmd: &Command) -> Option<Result<String, String>> {
    match kind {
        "dbase" | "dbf" => Some(fred::export_dbase(rest, ds, st, cmd)),
        _ => None,
    }
}
