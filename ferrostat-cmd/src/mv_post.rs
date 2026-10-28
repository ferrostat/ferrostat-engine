//! Dispatch of `predict` and `estat` after the multivariate commands, which
//! (unlike regression models) usually post no e(b)/e(V).

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::store::Store;

pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<Option<String>, String> {
    if let Some(out) = crate::mv_pca::predict(ds, st, cmd)? {
        return Ok(Some(out));
    }
    if let Some(out) = crate::mv_discrim::predict(ds, st, cmd)? {
        return Ok(Some(out));
    }
    Ok(None)
}

pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    if let Some(r) = crate::mv_pca::estat(ds, st, sub, cmd) {
        return Some(r);
    }
    if let Some(r) = crate::mv_mds::mds_estat(st, sub, cmd) {
        return Some(r);
    }
    if let Some(r) = crate::mv_mds::ca_estat(st, sub, cmd) {
        return Some(r);
    }
    if let Some(r) = crate::mv_discrim::estat(ds, st, sub, cmd) {
        return Some(r);
    }
    None
}
