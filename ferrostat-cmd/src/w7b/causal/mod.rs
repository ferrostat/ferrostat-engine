//! Treatment effects: endogenous-treatment regressions (`etregress`,
//! `etpoisson`), control-function effects (`eteffects`), heterogeneous
//! difference in differences (`hdidregress`, `xthdidregress`), switching
//! treatments (`xtswitchdid`), `gencohort`, and treatment effects after the
//! extended regression models (`estat teffects`, `predict` options).

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::store::Store;
use crate::xmodels::Handlers;

mod erm_post;
mod etreg;
mod eteffects;
mod hdid;
pub(crate) mod util;

pub fn handlers() -> Handlers {
    let mut v: Handlers = Vec::new();
    v.extend(etreg::handlers());
    v.extend(eteffects::handlers());
    v.extend(hdid::handlers());
    v
}

/// `predict` after this family's commands (None: not ours).
pub fn predict(ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Option<Result<String, String>> {
    if let Some(r) = erm_post::predict(ds, st, cmd) {
        return Some(r);
    }
    etreg::predict(ds, st, cmd)
}

/// `estat` after this family's commands (None: not ours).
pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    if let Some(r) = erm_post::estat(ds, st, sub, cmd) {
        return Some(r);
    }
    if let Some(r) = etreg::estat(ds, st, sub, cmd) {
        return Some(r);
    }
    hdid::estat(ds, st, sub, cmd)
}
