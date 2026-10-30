//! Choice models over long-format data (one row per case and alternative):
//! the `cmset` declaration, descriptive tools, conditional, rank-ordered,
//! mixed and nested logit, and multinomial / rank-ordered probit.

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::store::Store;
use crate::xmodels::Handlers;

pub mod data;
pub mod models;
pub mod nlogit;
pub mod tools;

pub fn handlers() -> Handlers {
    let mut v: Handlers = vec![
        ("cmset", Box::new(tools::Cmset)),
        ("cmtab", Box::new(tools::Cmtab)),
        ("cmsummarize", Box::new(tools::Cmsummarize)),
        ("cmchoiceset", Box::new(tools::Cmchoiceset)),
        ("cmsample", Box::new(tools::Cmsample)),
        ("nlogit", Box::new(nlogit::Nlogit)),
        ("nlogitgen", Box::new(nlogit::Nlogitgen)),
        ("nlogittree", Box::new(nlogit::Nlogittree)),
    ];
    for verb in models::VERBS {
        v.push((verb, Box::new(models::estimator(verb))));
    }
    v
}

/// `predict` after this family's commands (None: not ours).
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Option<Result<String, String>> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    if ecmd == "nlogit" {
        return Some(nlogit::predict(ds, st, cmd));
    }
    if models::VERBS.contains(&ecmd.as_str()) {
        return Some(models::predict(ds, st, cmd));
    }
    None
}

/// `estat` after this family's commands (None: not ours).
pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    if models::VERBS.contains(&ecmd.as_str()) {
        return models::estat(ds, st, sub, cmd);
    }
    None
}
