//! The `bayes` prefix: `bayes [, options] : estimation command`.
//!
//! The estimation command runs quietly (maximum likelihood); its results
//! are translated into a `bayesmh` model with default priors (see
//! `ferrostat_cmd::w7e::prefix`), which is then sampled and displayed.
//! `bayes` alone replays the last Bayesian results.

use ferrostat_cmd::eval::rc_err;
use ferrostat_parser::parser::find_top_level;

use crate::interpreter::{AdoInterpreter, Flow};
use crate::resample;

pub(crate) fn intercept(me: &mut AdoInterpreter, verb: &str, rest: &str) -> Result<Option<Flow>, String> {
    if verb == "bayesmh" && rest.contains("evaluator(") && !me.engine.store.programs.contains_key(verb) {
        bayesmh_evaluator(me, rest)?;
        return Ok(Some(Flow::Next));
    }
    if verb != "bayes" || me.engine.store.programs.contains_key(verb) {
        return Ok(None);
    }
    let r = rest.trim();
    let (opts, cmd) = if let Some(c) = r.strip_prefix(':') {
        (String::new(), c.trim().to_string())
    } else {
        match find_top_level(r, ':') {
            Some(i) => (r[..i].trim().trim_start_matches(',').trim().to_string(), r[i + 1..].trim().to_string()),
            None => {
                // replay
                if !ferrostat_cmd::w7e::is_bayes(&me.engine.store) {
                    return Err(rc_err(301, "last estimates not found"));
                }
                let out = me.engine.store.e.macros.get("_output").cloned().unwrap_or_default();
                me.emit(&out);
                return Ok(Some(Flow::Next));
            }
        }
    };
    if cmd.is_empty() {
        return Err(rc_err(198, "bayes: estimation command required"));
    }
    let spec = match ferrostat_cmd::w7e::prefix::direct(&cmd, &opts) {
        Some(s) => s,
        None => {
            // maximum-likelihood fit first: starting values and model details
            resample::run_cmd(me, &cmd, false)?;
            let eng = &mut me.engine;
            ferrostat_cmd::w7e::prefix::translate(&eng.dataset, &mut eng.store, &cmd, &opts)?
        }
    };
    let res = {
        let eng = &mut me.engine;
        ferrostat_cmd::w7e::run::estimate(&mut eng.dataset, &mut eng.store, &spec)
    };
    me.engine.store.matrices.remove("__bayes_qb");
    me.engine.store.matrices.remove("__bayes_qV");
    me.engine.store.matrices.remove("__bayes_Sigma");
    let out = res?;
    me.engine.store.e.set_str("cmdline", format!("bayes{}: {cmd}", if opts.is_empty() { String::new() } else { format!(", {opts}") }));
    me.emit(&out);
    Ok(Some(Flow::Next))
}

/// `bayesmh ..., llevaluator(prog [, parameters()])` / `evaluator(prog ...)`:
/// the log likelihood (log posterior) is computed by an ado program called
/// as `prog lnf xb1 [xb2 ...]`, where `lnf` is a temporary variable to fill
/// with the observation-level contributions (or a scalar of that name to
/// set to the total) and `xb#` hold the linear forms of the equations.
/// Globals: $MH_n (sample size), $MH_touse (sample marker), $MH_y1, ...
/// (outcomes), $MH_b (name of a row vector with the current parameters).
fn bayesmh_evaluator(me: &mut AdoInterpreter, rest: &str) -> Result<(), String> {
    use ferrostat_cmd::matrix::Matrix;
    use ferrostat_cmd::util;
    use ferrostat_core::missing::{is_missing, SYSMISS};
    let spec = ferrostat_cmd::w7e::bayesmh::parse(rest)?;
    let ev = spec.opts.iter().find(|(n, _)| n.starts_with("llev") || n.starts_with("eval")).and_then(|(_, a)| a.clone()).ok_or_else(|| rc_err(198, "evaluator() invalid"))?;
    let prog = ev.split(',').next().unwrap_or("").trim().to_string();
    if !me.engine.store.programs.contains_key(&prog) {
        return Err(rc_err(111, format!("program {prog} not found")));
    }
    for (k, e) in spec.eqs.iter().enumerate() {
        me.engine.store.globals.insert(format!("MH_y{}", k + 1), e.dep.clone());
    }
    me.engine.store.globals.insert("MH_b".into(), "MH_b".into());
    me.engine.store.globals.insert("MH_touse".into(), "__mh_touse".into());
    let neq = spec.eqs.len().max(1);
    let ptr: *mut AdoInterpreter = me;
    let cb = move |th: &[f64], names: &[String], xbs: &[Vec<f64>], rows: &[usize]| -> f64 {
        // SAFETY: the sampler works on copies of the data and results (see
        // below); the interpreter is not otherwise used while it runs.
        let me = unsafe { &mut *ptr };
        let n = me.engine.dataset.n_obs;
        me.engine.store.matrices.insert("MH_b".into(), Matrix::row_vector(th).with_names(vec!["y1".into()], names.to_vec()));
        me.engine.store.globals.insert("MH_n".into(), rows.len().to_string());
        let mut tmp = vec!["__mh_lnf".to_string(), "__mh_touse".to_string()];
        let mut put = |me: &mut AdoInterpreter, name: &str, vals: Vec<f64>| {
            let _ = me.engine.dataset.drop_variable(name);
            let _ = me.engine.dataset.add_variable(util::new_double(name, vals));
        };
        let mut touse = vec![0.0; n];
        for &i in rows {
            touse[i] = 1.0;
        }
        put(me, "__mh_touse", touse);
        put(me, "__mh_lnf", vec![SYSMISS; n]);
        let mut args = vec!["__mh_lnf".to_string()];
        for (k, xb) in xbs.iter().enumerate().take(neq) {
            let mut v = vec![SYSMISS; n];
            for (r, &i) in rows.iter().enumerate() {
                v[i] = xb[r];
            }
            let nm = format!("__mh_xb{}", k + 1);
            put(me, &nm, v);
            args.push(nm.clone());
            tmp.push(nm);
        }
        me.engine.store.scalars.remove("__mh_lnf");
        let ok = resample::run_cmd(me, &format!("{prog} {}", args.join(" ")), false).is_ok();
        let val = if !ok {
            f64::NEG_INFINITY
        } else if let Some(v) = me.engine.store.scalars.remove("__mh_lnf") {
            if is_missing(v) { f64::NEG_INFINITY } else { v }
        } else {
            let col = util::col(&me.engine.dataset, "__mh_lnf").unwrap_or_default();
            let mut s = 0.0;
            for &i in rows {
                let v = col.get(i).copied().unwrap_or(SYSMISS);
                if is_missing(v) {
                    s = f64::NEG_INFINITY;
                    break;
                }
                s += v;
            }
            s
        };
        for t in &tmp {
            let _ = me.engine.dataset.drop_variable(t);
        }
        val
    };
    ferrostat_cmd::w7e::model::CALLBACK.with(|c| *c.borrow_mut() = Some(ferrostat_cmd::w7e::model::Callback(std::rc::Rc::new(std::cell::RefCell::new(Box::new(cb))))));
    let mut ds2 = me.engine.dataset.clone();
    let mut st2 = me.engine.store.clone();
    let res = ferrostat_cmd::w7e::run::estimate(&mut ds2, &mut st2, rest);
    ferrostat_cmd::w7e::model::CALLBACK.with(|c| *c.borrow_mut() = None);
    let out = res?;
    me.engine.store.e = st2.e.clone();
    let rng = st2.rng.borrow().clone();
    *me.engine.store.rng.borrow_mut() = rng;
    me.engine.store.matrices.remove("MH_b");
    me.emit(&out);
    Ok(())
}
