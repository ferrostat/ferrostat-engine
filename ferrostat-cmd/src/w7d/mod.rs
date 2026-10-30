//! Wave 7d: meta-analysis (`meta`), Bayesian model averaging (`bma*`),
//! financial tools (`fin`) and DSGE models (`dsge`, `dsgenl`).

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::store::Store;
use crate::xmodels::Handlers;

pub mod bma;
pub mod dsge;
pub mod fin;
pub mod meta;
pub mod opt;

/// Commands implemented in this wave.
pub fn handlers() -> Handlers {
    let mut v: Handlers = Vec::new();
    v.push(("dsge", Box::new(dsge::Dsge { nonlinear: false })));
    v.push(("dsgenl", Box::new(dsge::Dsge { nonlinear: true })));
    v.push(("meta", Box::new(meta::Meta)));
    v.push(("fin", Box::new(fin::Fin)));
    v.extend(bma::handlers());
    v
}

/// `predict` after the commands of this wave (None: not ours).
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<Option<String>, String> {
    if meta::reg::is_meta_regress(st) {
        return meta::reg::predict(ds, st, cmd).map(Some);
    }
    Ok(None)
}

/// `estat` after the commands of this wave (None: not ours).
pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    if let Some(r) = meta::reg::estat(ds, st, sub, cmd) {
        return Some(r);
    }
    dsge::estat(ds, st, sub, cmd)
}
