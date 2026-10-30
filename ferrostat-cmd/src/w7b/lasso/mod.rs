//! Lasso, elastic net and square-root lasso for linear, logit, probit and
//! Poisson models (`core`: coordinate descent, cross-validation, plug-in
//! penalty; `fit`: selection and stored paths; `cmd`: the estimation
//! commands; `post`: postestimation; `inf`: inference on variables of
//! interest after selection of controls — double selection,
//! partialling-out and cross-fit partialling-out).

pub mod cmd;
pub mod core;
pub mod fit;
pub mod inf;
pub mod post;

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::store::Store;
use crate::xmodels::Handlers;

pub fn handlers() -> Handlers {
    let mut v: Handlers = Vec::new();
    for verb in ["lasso", "elasticnet", "sqrtlasso"] {
        v.push((verb, Box::new(cmd::LassoCmd { verb })));
    }
    for verb in ["lassoinfo", "lassoknots", "lassocoef", "lassogof", "lassoselect", "cvplot", "coefpath", "bicplot"] {
        v.push((verb, Box::new(post::Post { verb })));
    }
    for verb in inf::VERBS {
        v.push((verb, Box::new(inf::Inf { verb })));
    }
    v
}

/// `predict` after lasso, elasticnet and sqrtlasso (None: not ours).
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Option<Result<String, String>> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    if matches!(ecmd.as_str(), "lasso" | "elasticnet" | "sqrtlasso") {
        return Some(post::predict(ds, st, cmd));
    }
    if inf::VERBS.contains(&ecmd.as_str()) {
        return Some(Err(crate::eval::rc_err(321, format!("predict not available after {ecmd}"))));
    }
    None
}

/// `estat` after lasso fits: none beyond the generic ones.
pub fn estat(_ds: &mut DataSet, st: &mut Store, sub: &str, _cmd: &Command) -> Option<Result<String, String>> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    if matches!(ecmd.as_str(), "lasso" | "elasticnet" | "sqrtlasso") && sub != "summarize" {
        return Some(Err(crate::eval::rc_err(321, format!("estat {sub} not available after {ecmd}"))));
    }
    None
}

/// A numlist (`0.5 0.75 1`, `0.1(0.1)1`).
pub fn numlist(text: &str) -> Result<Vec<f64>, String> {
    crate::margins::numlist(text)
}
