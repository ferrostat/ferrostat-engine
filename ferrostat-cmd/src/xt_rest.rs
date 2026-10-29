//! Registration of the remaining panel, survival and multilevel commands
//! (panel cointegration tests, extended regression models, panel
//! multinomial logit, panel selection, panel VAR, survival utilities, menl).

use crate::commands::CommandHandler;

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    let mut v: Vec<(&'static str, Box<dyn CommandHandler>)> = vec![("xtcointtest", Box::new(crate::xt_coint::Xtcointtest))];
    v.extend(crate::erm::handlers());
    v.extend(crate::xt_mvar::handlers());
    v.extend(crate::st_rest::handlers());
    v.extend(crate::menl::handlers());
    v
}

/// `predict` after the commands registered here (None: not ours).
pub fn predict(ds: &mut ferrostat_core::DataSet, st: &mut crate::store::Store, cmd: &ferrostat_parser::Command) -> Option<Result<String, String>> {
    crate::erm::predict(ds, st, cmd)
}
