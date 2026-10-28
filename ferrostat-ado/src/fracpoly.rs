//! Fractional polynomials: `fp` and `mfp`.

use crate::interpreter::{AdoInterpreter, Flow};

pub(crate) fn intercept(_me: &mut AdoInterpreter, _verb: &str, _rest: &str) -> Result<Option<Flow>, String> {
    Ok(None)
}
