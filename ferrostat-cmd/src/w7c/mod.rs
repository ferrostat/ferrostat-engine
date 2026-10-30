//! Wave 7c: structural equation models (`sem`, `gsem`), item response
//! theory (`irt`, `dif*`) and finite mixture models (`fmm`).
//!
//! Shared pieces: path notation ([`path`]), the linear model in
//! reticular-action form ([`model`]), numerical optimization ([`opt`]),
//! the generalized-response engine with adaptive quadrature and latent
//! classes ([`gsem`]), and a registry of fitted models for postestimation
//! ([`state`]).

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::store::Store;
use crate::xmodels::Handlers;

pub mod fmm;
pub mod engine;
pub mod gsem;
pub mod irt;
pub mod la;
pub mod model;
pub mod opt;
pub mod path;
pub mod sem;
pub mod sempost;
pub mod ssd;
pub mod state;
pub mod tab;

pub fn handlers() -> Handlers {
    let mut v: Handlers = vec![("sem", Box::new(sem::Sem)), ("ssd", Box::new(ssd::Ssd))];
    v.extend(gsem::handlers());
    v.extend(fmm::handlers());
    v
}

/// `predict` after the commands of this wave (None: not ours).
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<Option<String>, String> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    match ecmd.as_str() {
        "sem" => sempost::predict(ds, st, cmd).map(Some),
        "gsem" | "irt" => gsem::predict(ds, st, cmd).map(Some),
        "fmm" => fmm::predict(ds, st, cmd).map(Some),
        _ => Ok(None),
    }
}

/// `estat` after the commands of this wave (None: not ours).
pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    match ecmd.as_str() {
        "sem" => sempost::estat(ds, st, sub, cmd),
        "gsem" | "irt" => gsem::estat(ds, st, sub, cmd),
        "fmm" => fmm::estat(ds, st, sub, cmd),
        _ => None,
    }
}
