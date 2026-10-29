//! Registration and postestimation dispatch for the additional time-series
//! commands (arch, mgarch, arfima, sspace, dfactor, ucm, mswitch,
//! threshold, forecast, lpirf, tsfill, tsappend, ...).

use ferrostat_core::DataSet;
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::store::Store;

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    vec![
        ("arch", Box::new(crate::arch::Arch)),
        ("sspace", Box::new(crate::sspace::Sspace)),
        ("constraint", Box::new(crate::sspace::Constraint)),
        ("threshold", Box::new(crate::threshold::Threshold)),
        ("mswitch", Box::new(crate::mswitch::Mswitch)),
        ("arfima", Box::new(crate::arfima::Arfima)),
        ("arimasoc", Box::new(crate::arfima::Soc { fractional: false })),
        ("arfimasoc", Box::new(crate::arfima::Soc { fractional: true })),
        ("forecast", Box::new(crate::forecast::Forecast)),
        ("lpirf", Box::new(crate::lpirf::Lpirf { iv: false })),
        ("ivlpirf", Box::new(crate::lpirf::Lpirf { iv: true })),
        ("mgarch", Box::new(crate::mgarch::Mgarch)),
        ("ucm", Box::new(crate::ts_ucm::UcmCmd)),
        ("dfactor", Box::new(crate::ts_ucm::DfactorCmd)),
        ("tsfill", Box::new(crate::tsutil::Tsfill)),
        ("tsappend", Box::new(crate::tsutil::Tsappend)),
        ("tsreport", Box::new(crate::tsutil::Tsreport)),
        ("tsrevar", Box::new(crate::tsutil::Tsrevar)),
        ("psdensity", Box::new(crate::tsutil::Psdensity)),
    ]
}

/// `predict` after one of these estimators (None if not ours).
pub fn predict(ds: &mut DataSet, st: &Store, cmd: &Command) -> Option<Result<String, String>> {
    let ecmd = st.e.macros.get("cmd").cloned().unwrap_or_default();
    match ecmd.as_str() {
        "arch" => Some(crate::arch::predict(ds, st, cmd)),
        "threshold" => Some(crate::threshold::predict(ds, st, cmd)),
        "mswitch" => Some(crate::mswitch::predict(ds, st, cmd)),
        "arfima" => Some(crate::arfima::predict(ds, st, cmd)),
        "mgarch" => Some(crate::mgarch::predict(ds, st, cmd)),
        "ucm" | "dfactor" => Some(crate::ts_ucm::predict(ds, st, cmd)),
        "sspace" => Some(crate::sspace::predict(ds, st, cmd)),
        _ => None,
    }
}

/// `estat` subcommands after these estimators (None if not ours).
pub fn estat(ds: &mut DataSet, st: &mut Store, sub: &str, cmd: &Command) -> Option<Result<String, String>> {
    if let Some(r) = crate::tssb::estat(ds, st, sub, cmd) {
        return Some(r);
    }
    if let Some(r) = crate::mswitch::estat(st, sub) {
        return Some(r);
    }
    None
}
