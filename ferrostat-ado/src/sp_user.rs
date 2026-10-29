//! `spmatrix userdefined Wname = fcn(varlist) [if] [in] [, normalize() replace]`:
//! a spatial weighting matrix whose (i,j) element is the value of the
//! user's Mata function `fcn(vi, vj)`, where `vi` and `vj` are the row
//! vectors of `varlist` for spatial units i and j (the diagonal is zero).

use ferrostat_cmd::eval::rc_err;
use ferrostat_cmd::util;
use ferrostat_core::missing::is_missing;
use ferrostat_parser::parser::parse_line;

use crate::interpreter::{AdoInterpreter, Flow};
use crate::resample::run_cmd;

pub(crate) fn intercept(me: &mut AdoInterpreter, verb_raw: &str, rest: &str, _text: &str) -> Result<Option<Flow>, String> {
    if !verb_raw.eq_ignore_ascii_case("spmatrix") || me.engine.store.programs.contains_key(verb_raw) {
        return Ok(None);
    }
    let r = rest.trim();
    let sub = r.split_whitespace().next().unwrap_or("");
    if !sub.eq_ignore_ascii_case("userdefined") {
        return Ok(None);
    }
    userdefined(me, r[sub.len()..].trim())?;
    Ok(Some(Flow::Next))
}

fn userdefined(me: &mut AdoInterpreter, args: &str) -> Result<(), String> {
    // Wname = fcn(varlist) [if] [in] [, options]
    let (name, rhs) = args.split_once('=').ok_or_else(|| rc_err(198, "spmatrix userdefined Wname = fcn(varlist)"))?;
    let name = name.trim().to_string();
    let rhs = rhs.trim();
    let open = rhs.find('(').ok_or_else(|| rc_err(198, "function(varlist) required"))?;
    let close = rhs.find(')').ok_or_else(|| rc_err(198, "unmatched parenthesis"))?;
    let fcn = rhs[..open].trim().to_string();
    if fcn.is_empty() || !fcn.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return Err(rc_err(198, format!("invalid function name {fcn}")));
    }
    let vlist = rhs[open + 1..close].trim().to_string();
    let tail = rhs[close + 1..].to_string();
    // the remainder parses like an ordinary command (if/in/options)
    let cmd = parse_line(&format!("spmatrix_ud {vlist} {tail}")).map_err(|e| rc_err(198, e.message))?;
    let ds = &me.engine.dataset;
    let s = ferrostat_cmd::sp_data::need_spset(ds)?;
    let vars = ferrostat_cmd::varlist::expand(ds, &vlist)?;
    let mut mask = util::touse(ds, &me.engine.store, &cmd)?;
    util::markout(ds, &mut mask, &vars);
    let ids = util::col(ds, &s.id)?;
    let mut seen = std::collections::HashSet::new();
    for i in 0..ds.n_obs {
        if mask[i] && (is_missing(ids[i]) || !seen.insert(ids[i].to_bits())) {
            mask[i] = false;
        }
    }
    if !mask.iter().any(|&b| b) {
        return Err(rc_err(2000, "no observations"));
    }
    let tu = "__spud_touse";
    let _ = me.engine.dataset.drop_variable(tu);
    me.engine.dataset.add_variable(util::new_double(tu, mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect()))?;
    let mut opts: Vec<String> = Vec::new();
    for (k, v) in &cmd.opts {
        opts.push(match v {
            Some(a) => format!("{k}({a})"),
            None => k.clone(),
        });
    }
    let line = format!(
        "mata: __spud_X = st_data(., \"{}\", \"{tu}\"); __spud_I = st_data(., \"{}\", \"{tu}\"); __spud_n = rows(__spud_X); __spud_W = J(__spud_n, __spud_n, 0); for (__spud_i=1; __spud_i<=__spud_n; __spud_i++) {{ for (__spud_j=1; __spud_j<=__spud_n; __spud_j++) {{ if (__spud_i != __spud_j) __spud_W[__spud_i,__spud_j] = {fcn}(__spud_X[__spud_i,.], __spud_X[__spud_j,.]) }} }}; st_matrix(\"__spud_W\", __spud_W); st_matrix(\"__spud_I\", __spud_I)",
        vars.join(" "),
        s.id
    );
    let r1 = run_cmd(me, &line, false);
    let _ = me.engine.dataset.drop_variable(tu);
    let _ = run_cmd(me, "mata: mata drop __spud_X __spud_I __spud_n __spud_W __spud_i __spud_j", false);
    r1?;
    let r2 = run_cmd(me, &format!("spmatrix spfrommata {name} = __spud_W __spud_I{}", if opts.is_empty() { String::new() } else { format!(", {}", opts.join(" ")) }), true);
    me.engine.store.matrices.remove("__spud_W");
    me.engine.store.matrices.remove("__spud_I");
    r2
}
