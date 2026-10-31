//! Wave 7e: Bayesian analysis.
//!
//! `bayesmh` (adaptive Metropolis-Hastings with Gibbs updates for
//! conjugate blocks), the `bayes:` prefix (translated to a `bayesmh`
//! model by `prefix`), and the postestimation commands `bayesstats`,
//! `bayestest`, `bayesgraph`, `bayespredict`, `bayesreps`, `bayesirf`,
//! `bayesfcast`, `bayesvarstable` and `bayesselect`.

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::store::Store;
use crate::xmodels::Handlers;

pub mod bayesmh;
pub mod expr;
pub mod mcmc;
pub mod model;
pub mod post;
pub mod predict;
pub mod select;
pub mod var;
pub mod prefix;
pub mod prior;
pub mod run;
pub mod stats;
pub mod text;

pub struct Bayes {
    verb: &'static str,
}

/// Text after the verb of a command line.
pub fn after_verb(cmd: &Command) -> String {
    let raw = cmd.raw.trim();
    let i = raw.find(|c: char| c.is_whitespace() || c == ',').unwrap_or(raw.len());
    raw[i..].trim().to_string()
}

impl CommandHandler for Bayes {
    fn name(&self) -> &'static str {
        self.verb
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let rest = after_verb(cmd);
        match self.verb {
            "bayesmh" => {
                if rest.trim().is_empty() || rest.trim_start().starts_with(',') && !rest.contains("likelihood") && !rest.contains("prior") {
                    return post::replay(ds, st, &rest);
                }
                run::estimate(ds, st, &rest)
            }
            v => post::dispatch(v, ds, st, &rest, cmd),
        }
    }
}

pub fn handlers() -> Handlers {
    let mut v: Handlers = Vec::new();
    for n in ["bayesmh", "bayesstats", "bayestest", "bayesgraph", "bayespredict", "bayesreps", "bayesirf", "bayesfcast", "bayesvarstable", "bayesselect"] {
        v.push((n, Box::new(Bayes { verb: n })));
    }
    v
}

/// Is the current estimation result Bayesian?
pub fn is_bayes(st: &Store) -> bool {
    st.e.matrices.contains_key("_mcmc")
}

/// `estat` after Bayesian estimation (None: not ours).
pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    if !is_bayes(st) {
        return None;
    }
    post::estat(ds, st, sub, cmd)
}
