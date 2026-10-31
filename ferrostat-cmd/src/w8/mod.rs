//! Wave 8: the remaining base-reference items (display and random-number
//! settings, ROC regression, demand systems, programmer's helpers).

pub mod display;
pub mod rngkind;
pub mod rocreg;
pub mod demandsys;
pub mod robust;

use crate::xmodels::Handlers;

/// Commands implemented in this wave.
pub fn handlers() -> Handlers {
    let mut v: Handlers = Vec::new();
    v.push(("rocreg", Box::new(rocreg::Rocreg)));
    v.push(("rocregplot", Box::new(rocreg::Rocregplot)));
    v.push(("demandsys", Box::new(demandsys::Demandsys)));
    v.push(("_robust", Box::new(robust::Robust)));
    v
}

use crate::store::Store;

/// `set` for the settings of this wave; None when `what` is not one of them.
pub fn set(st: &mut Store, what: &str, v: &str) -> Option<Result<String, String>> {
    let v = v.split(',').next().unwrap_or("").trim();
    let r = match what {
        "cformat" | "pformat" | "sformat" => display::check_format(what, v),
        "showbaselevels" | "emptycells" => display::check_choice(what, v),
        "rng" => rngkind::set_kind(&mut st.rng.borrow_mut(), v),
        "rngstream" => rngkind::set_stream(&mut st.rng.borrow_mut(), v),
        "rngstate" => rngkind::set_state(&mut st.rng.borrow_mut(), v),
        _ => return None,
    };
    if let Err(e) = r {
        return Some(Err(e));
    }
    match what {
        "rngstate" => {
            let k = st.rng.borrow().kind;
            st.settings.insert("rngstream".into(), st.rng.borrow().stream.to_string());
            if k != rngkind::Kind::Mt64 || st.settings.get("rng").is_some_and(|x| x != "default") {
                st.settings.insert("rng".into(), k.name().into());
            }
        }
        "rngstream" => {
            st.settings.insert(what.into(), v.trim().parse::<u64>().unwrap_or(1).to_string());
        }
        _ if v.is_empty() => {
            st.settings.remove(what);
        }
        _ => {
            st.settings.insert(what.into(), v.into());
        }
    }
    display::sync(st);
    Some(Ok(String::new()))
}

/// `estat` after the commands of this wave (None: not ours).
pub fn estat(ds: &mut ferrostat_core::DataSet, st: &mut Store, sub: &str, cmd: &ferrostat_parser::Command) -> Option<Result<String, String>> {
    demandsys::estat(ds, st, sub, cmd)
}

/// `predict` after the commands of this wave (None: not ours).
pub fn predict(ds: &mut ferrostat_core::DataSet, st: &mut Store, cmd: &ferrostat_parser::Command) -> Result<Option<String>, String> {
    demandsys::predict(ds, st, cmd).transpose()
}
