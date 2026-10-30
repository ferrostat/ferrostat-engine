//! Wave 7b: penalized regression and inference after variable selection
//! (`lasso`), discrete-choice models over choice sets (`cm`), and the
//! remaining treatment-effect estimators (`causal`).
//!
//! Each family exposes its command list and optional `predict` / `estat`
//! hooks; this module only gathers them.

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::store::Store;
use crate::xmodels::Handlers;

pub mod causal;
pub mod causal2;
pub mod cm;
pub mod lasso;

/// Commands implemented in this wave.
pub fn handlers() -> Handlers {
    let mut v: Handlers = Vec::new();
    v.extend(lasso::handlers());
    v.extend(cm::handlers());
    v.extend(causal::handlers());
    v.extend(causal2::handlers());
    v
}

/// `predict` after the commands of this wave (None: not ours).
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<Option<String>, String> {
    if let Some(r) = lasso::predict(ds, st, cmd) {
        return r.map(Some);
    }
    if let Some(r) = cm::predict(ds, st, cmd) {
        return r.map(Some);
    }
    if let Some(r) = causal::predict(ds, st, cmd) {
        return r.map(Some);
    }
    if let Some(r) = causal2::predict(ds, st, cmd) {
        return r.map(Some);
    }
    Ok(None)
}

/// `estat` after the commands of this wave (None: not ours).
pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    if let Some(r) = lasso::estat(ds, st, sub, cmd) {
        return Some(r);
    }
    if let Some(r) = cm::estat(ds, st, sub, cmd) {
        return Some(r);
    }
    if let Some(r) = causal::estat(ds, st, sub, cmd) {
        return Some(r);
    }
    causal2::estat(ds, st, sub, cmd)
}
