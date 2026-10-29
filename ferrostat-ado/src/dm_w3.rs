//! Data-management commands that need the interpreter: `dyngen` blocks
//! (simultaneous, observation-by-observation updates) and `putmata` /
//! `getmata` (copying variables to and from the matrix language).

use ferrostat_cmd::eval::{rc_err, Env};
use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::VarType;
use ferrostat_parser::parser::{parse_expression, parse_line};
use ferrostat_parser::Expr;

use crate::interpreter::{AdoInterpreter, Flow};
use crate::script::Stmt;

struct Update {
    var: String,
    expr: Expr,
    cond: Option<Expr>,
}

/// Replace every reference to a variable `v` (plain or subscripted) with
/// `cond(missing(v), mv, v)`.
fn with_missval(e: Expr, mv: f64, is_var: &dyn Fn(&str) -> bool) -> Expr {
    let wrap = |node: Expr| Expr::Call("cond".into(), vec![Expr::Call("missing".into(), vec![node.clone()]), Expr::Number(mv), node]);
    match e {
        Expr::Var(ref n) if is_var(n) => wrap(e),
        Expr::Index(base, sub) => {
            let sub = with_missval(*sub, mv, is_var);
            let node = Expr::Index(base, Box::new(sub));
            match &node {
                Expr::Index(b, _) if matches!(b.as_ref(), Expr::Var(n) if is_var(n)) => wrap(node),
                _ => node,
            }
        }
        Expr::Binary(a, op, b) => Expr::Binary(Box::new(with_missval(*a, mv, is_var)), op, Box::new(with_missval(*b, mv, is_var))),
        Expr::Unary(op, a) => Expr::Unary(op, Box::new(with_missval(*a, mv, is_var))),
        Expr::Paren(a) => Expr::Paren(Box::new(with_missval(*a, mv, is_var))),
        Expr::Call(f, args) => Expr::Call(f, args.into_iter().map(|a| with_missval(a, mv, is_var)).collect()),
        other => other,
    }
}

/// `dyngen [if] [in] { update var = exp [if] [, missval(#)] ... }`
pub fn dyngen(me: &mut AdoInterpreter, header: &str, body: &[Stmt]) -> Result<Flow, String> {
    let header = header.trim().trim_end_matches('{').trim();
    let hcmd = parse_line(&format!("dyngen {header}")).map_err(|e| rc_err(198, format!("{e:?}")))?;
    let mut updates = Vec::new();
    for s in body {
        let text = {
            let eng = &mut me.engine;
            crate::macros::expand(s.text.trim(), &eng.dataset, &mut eng.store)?
        };
        let text = text.trim();
        if text.is_empty() || text.starts_with('*') || text.starts_with("//") {
            continue;
        }
        if !text.starts_with("update ") {
            return Err(rc_err(198, format!("dyngen: only update statements are allowed; found \"{text}\"")));
        }
        let c = parse_line(text).map_err(|e| rc_err(198, format!("{e:?}")))?;
        let (lhs, rhs) = c.main.split_once('=').ok_or_else(|| rc_err(198, "update varname = exp"))?;
        let var = ferrostat_cmd::varlist::resolve_name(&me.engine.dataset, lhs.trim())?;
        let v = me.engine.dataset.get_var(&var).unwrap();
        if v.var_type.is_string() {
            return Err(rc_err(109, format!("{var}: numeric variable required")));
        }
        let mut expr = parse_expression(rhs.trim()).map_err(|e| rc_err(198, format!("invalid expression: {e:?}")))?;
        if let Some(mv) = c.opt("missval", 1) {
            let mv: f64 = mv.trim().parse().map_err(|_| rc_err(198, "missval() requires a number"))?;
            let ds = &me.engine.dataset;
            let is_var = |n: &str| !n.starts_with('_') && ds.get_var(n).is_some_and(|v| !v.var_type.is_string());
            expr = with_missval(expr, mv, &is_var);
        }
        updates.push(Update { var, expr, cond: c.if_expr.clone() });
    }
    if updates.is_empty() {
        return Err(rc_err(198, "dyngen: no update statements"));
    }
    let eng = &mut me.engine;
    let mask = ferrostat_cmd::util::touse(&eng.dataset, &eng.store, &hcmd)?;
    let mut changes = vec![0usize; updates.len()];
    for row in 0..eng.dataset.n_obs {
        if !mask[row] {
            continue;
        }
        for (k, u) in updates.iter().enumerate() {
            let val = {
                let env = Env::with_store(&eng.dataset, &eng.store);
                if let Some(c) = &u.cond {
                    let t = env.eval_num(c, row)?;
                    if t == 0.0 || is_missing(t) {
                        continue;
                    }
                }
                env.eval_num(&u.expr, row)?
            };
            let v = eng.dataset.get_var_mut(&u.var).unwrap();
            let ty = v.var_type;
            let old = v.num(row);
            let stored = ferrostat_cmd::util::typed_value(ty, if val >= SYSMISS { SYSMISS } else { val });
            v.set(row, stored);
            let new = v.num(row);
            if old.to_bits() != new.to_bits() && !(is_missing(old) && is_missing(new)) {
                changes[k] += 1;
            }
        }
    }
    eng.store.data_changed = true;
    let mut out = Vec::new();
    for (k, u) in updates.iter().enumerate() {
        out.push(format!("({} real change{} made to {})", changes[k], if changes[k] == 1 { "" } else { "s" }, u.var));
    }
    me.emit(&out.join("\n"));
    Ok(Flow::Next)
}

/// One `putmata` target: matrix name and source variables.
fn putmata_items(main: &str) -> Result<Vec<(String, Vec<String>)>, String> {
    let mut items = Vec::new();
    let chars: Vec<char> = main.chars().collect();
    let mut i = 0;
    let mut words: Vec<String> = Vec::new();
    while i < chars.len() {
        if chars[i] == '(' {
            let close = chars[i..].iter().position(|&c| c == ')').ok_or_else(|| rc_err(198, "unmatched parenthesis"))? + i;
            words.push(chars[i..=close].iter().collect());
            i = close + 1;
        } else if chars[i].is_whitespace() {
            i += 1;
        } else {
            let start = i;
            while i < chars.len() && !chars[i].is_whitespace() && chars[i] != '(' {
                i += 1;
            }
            words.push(chars[start..i].iter().collect());
        }
    }
    // normalize `name = var` spacing inside and outside parentheses
    let text = words.join(" ");
    let text = text.replace(" = ", "=").replace(" =", "=").replace("= ", "=");
    let mut toks: Vec<String> = Vec::new();
    let (mut cur, mut depth) = (String::new(), 0);
    for c in text.chars() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            c if c.is_whitespace() && depth == 0 => {
                if !cur.is_empty() {
                    toks.push(std::mem::take(&mut cur));
                }
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    if !cur.is_empty() {
        toks.push(cur);
    }
    for t in toks {
        let t: String = t.chars().filter(|&c| c != '(' && c != ')').collect();
        match t.split_once('=') {
            Some((name, vars)) => items.push((name.trim().to_string(), vars.split_whitespace().map(String::from).collect())),
            None => items.push((t.clone(), vec![t])),
        }
    }
    Ok(items)
}

/// `putmata [name=]varname ... [(name = varlist)] [if] [in] [, omitmissing replace view]`
pub fn putmata(me: &mut AdoInterpreter, rest: &str) -> Result<String, String> {
    let cmd = parse_line(&format!("putmata {rest}")).map_err(|e| rc_err(198, format!("{e:?}")))?;
    let ds = &me.engine.dataset;
    let mut items = putmata_items(&cmd.main)?;
    if items.is_empty() || items.len() == 1 && (items[0].0 == "*" || items[0].0 == "_all") {
        items = ds.vars.iter().map(|v| (v.name.clone(), vec![v.name.clone()])).collect();
    }
    let mut resolved = Vec::new();
    for (name, vars) in &items {
        let mut vs = Vec::new();
        for v in vars {
            vs.extend(ferrostat_cmd::varlist::expand(ds, v)?);
        }
        let strs = vs.iter().filter(|v| ds.get_var(v).unwrap().var_type.is_string()).count();
        if strs > 0 && strs < vs.len() {
            return Err(rc_err(109, format!("{name}: string and numeric variables may not be combined")));
        }
        resolved.push((name.clone(), vs, strs > 0));
    }
    let mut mask = ferrostat_cmd::util::touse(ds, &me.engine.store, &cmd)?;
    if cmd.has_opt("omitmissing", 4) {
        let all: Vec<String> = resolved.iter().flat_map(|r| r.1.clone()).collect();
        ferrostat_cmd::util::markout(ds, &mut mask, &all);
    }
    let tmp = "__putmata_touse";
    let mut tv = ferrostat_core::Variable::new(tmp, VarType::Byte);
    for &m in &mask {
        tv.push(ferrostat_core::Value::Byte(m as i8));
    }
    let changed = me.engine.store.data_changed;
    me.engine.dataset.add_variable(tv).map_err(|e| rc_err(110, e))?;
    let mut res = Ok(());
    let mut out = Vec::new();
    for (name, vars, is_str) in &resolved {
        ferrostat_cmd::varlist::check_name(name)?;
        let f = if *is_str { "st_sdata" } else { "st_data" };
        let line = format!("{name} = {f}(., \"{}\", \"{tmp}\")", vars.join(" "));
        res = crate::mata::run_line(me, &line).map(|_| ());
        if res.is_err() {
            break;
        }
        out.push(name.clone());
    }
    let _ = me.engine.dataset.drop_variable(tmp);
    me.engine.store.data_changed = changed;
    res?;
    let n = mask.iter().filter(|&&b| b).count();
    Ok(format!("({} matri{} posted, {n} row{}: {})", out.len(), if out.len() == 1 { "x" } else { "ces" }, if n == 1 { "" } else { "s" }, out.join(" ")))
}

/// `getmata varname[=matname] ... | (v1 v2 ...)=matname [, replace update double force]`
pub fn getmata(me: &mut AdoInterpreter, rest: &str) -> Result<String, String> {
    let cmd = parse_line(&format!("getmata {rest}")).map_err(|e| rc_err(198, format!("{e:?}")))?;
    let items = putmata_items(&cmd.main)?;
    if items.is_empty() {
        return Err(rc_err(198, "nothing to get"));
    }
    let replace = cmd.has_opt("replace", 3) || cmd.has_opt("update", 3);
    let force = cmd.has_opt("force", 5);
    let double = cmd.has_opt("double", 3);
    let mut made = Vec::new();
    for (lhs, rhs) in items {
        // `(a b)=M` arrives as name "a b" with vars ["M"]; `a=M` as ("a", ["M"]); `a` as ("a", ["a"])
        let (vars, mat): (Vec<String>, String) = (lhs.split_whitespace().map(String::from).collect(), rhs.join(" "));
        let mat = mat.trim().to_string();
        crate::mata::run_line(me, &format!("st_numscalar(\"__gm_r\", rows({mat})); st_numscalar(\"__gm_c\", cols({mat})); st_strscalar(\"__gm_t\", eltype({mat}))"))?;
        let st = &mut me.engine.store;
        let rows = st.scalars.remove("__gm_r").unwrap_or(0.0) as usize;
        let cols = st.scalars.remove("__gm_c").unwrap_or(0.0) as usize;
        let ty = st.str_scalars.remove("__gm_t").unwrap_or_default();
        if cols != vars.len() {
            return Err(rc_err(503, format!("{mat} has {cols} column{}, {} variable{} specified", if cols == 1 { "" } else { "s" }, vars.len(), if vars.len() == 1 { "" } else { "s" })));
        }
        let n = me.engine.dataset.n_obs;
        if rows != n && !force {
            if n == 0 && me.engine.dataset.n_vars() == 0 {
                // empty dataset: create the observations
                ferrostat_cmd::data::set_obs(&mut me.engine.dataset, rows)?;
            } else {
                return Err(rc_err(3200, format!("{mat} has {rows} rows but there are {n} observations; specify force")));
            }
        } else if rows > n {
            ferrostat_cmd::data::set_obs(&mut me.engine.dataset, rows)?;
        }
        let is_str = ty == "string";
        for (j, v) in vars.iter().enumerate() {
            let exists = me.engine.dataset.get_var(v).is_some();
            if exists && !replace {
                return Err(rc_err(110, format!("variable {v} already defined; specify replace")));
            }
            if !exists {
                ferrostat_cmd::varlist::check_name(v)?;
                let t = if is_str { VarType::Str(1) } else if double { VarType::Double } else { VarType::Float };
                let mut nv = ferrostat_core::Variable::new(v.as_str(), t);
                nv.format = if is_str { "%9s".into() } else if double { "%10.0g".into() } else { "%9.0g".into() };
                me.engine.dataset.add_variable(nv).map_err(|e| rc_err(110, e))?;
            }
            let f = if is_str { "st_sstore" } else { "st_store" };
            crate::mata::run_line(me, &format!("{f}((1::{rows}), \"{v}\", {mat}[., {}])", j + 1))?;
            made.push(v.clone());
        }
    }
    me.engine.store.data_changed = true;
    Ok(format!("({} variable{} created or replaced: {})", made.len(), if made.len() == 1 { "" } else { "s" }, made.join(" ")))
}
