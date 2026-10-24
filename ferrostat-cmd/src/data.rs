//! Creating and modifying variables and observations.

use ferrostat_core::missing::{is_missing, SYSMISS};
use ferrostat_core::{DataSet, Value, VarType, Variable};
use ferrostat_parser::parser::{find_top_level, parse_expression};
use ferrostat_parser::Command;

use crate::commands::CommandHandler;
use crate::eval::{rc_err, Env, Val};
use crate::store::Store;
use crate::util::{self, best_numeric_type, fits, promote, typed_value};
use crate::varlist;

fn plural(n: usize, word: &str) -> String {
    format!("{} {word}{}", crate::util::commas(n), if n == 1 { "" } else { "s" })
}

/// Split `[type] name = exp` from the main text of generate/replace.
fn split_assignment(cmd: &Command) -> Result<(String, String), String> {
    let main = &cmd.main;
    let eq = find_top_level(main, '=').ok_or_else(|| rc_err(198, "= required"))?;
    let lhs = main[..eq].trim().to_string();
    let rhs = main[eq + 1..].trim().to_string();
    let rhs = rhs.strip_prefix('=').map(|s| s.trim().to_string()).unwrap_or(rhs);
    if rhs.is_empty() {
        return Err(rc_err(198, "invalid syntax"));
    }
    Ok((lhs, rhs))
}

/// Evaluate an expression over the selected observations.
pub fn eval_over(ds: &DataSet, st: &Store, expr: &str, mask: &[bool]) -> Result<Vec<Option<Val>>, String> {
    let e = parse_expression(expr).map_err(|e| rc_err(198, e.message))?;
    let env = Env::with_store(ds, st).groups(st.by_groups.as_ref());
    let mut out = Vec::with_capacity(ds.n_obs);
    for i in 0..ds.n_obs {
        out.push(if mask[i] { Some(env.eval(&e, i)?) } else { None });
    }
    Ok(out)
}

pub struct Generate;

impl CommandHandler for Generate {
    fn name(&self) -> &'static str {
        "generate"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (lhs, rhs) = split_assignment(cmd)?;
        let parts: Vec<&str> = lhs.split_whitespace().collect();
        let (ty, name) = match parts.as_slice() {
            [n] => (None, *n),
            [t, n] => (Some(parse_type(t)?), *n),
            _ => return Err(rc_err(198, "invalid syntax")),
        };
        let (name, lblname) = match name.split_once(':') {
            Some((a, b)) => (a, Some(b.to_string())),
            None => (name, None),
        };
        varlist::check_name(name)?;
        if ds.var_index.contains_key(name) {
            return Err(rc_err(110, format!("variable {name} already defined")));
        }
        let mask = util::touse(ds, st, cmd)?;
        let vals = eval_over(ds, st, &rhs, &mask)?;
        let is_str = vals.iter().flatten().next().map_or_else(
            || {
                // no observation selected: decide by evaluating type on an empty row
                matches!(ty, Some(VarType::Str(_)) | Some(VarType::StrL))
                    || rhs.trim_start().starts_with('"')
            },
            |v| matches!(v, Val::Str(_)),
        );
        let mut var;
        let mut nmiss = 0;
        if is_str {
            if matches!(ty, Some(t) if t.is_numeric()) {
                return Err(rc_err(109, "type mismatch"));
            }
            let strs: Vec<String> = vals
                .iter()
                .map(|v| match v {
                    Some(Val::Str(s)) => Ok(s.clone()),
                    Some(Val::Num(_)) => Err(rc_err(109, "type mismatch")),
                    None => Ok(String::new()),
                })
                .collect::<Result<_, _>>()?;
            let maxlen = strs.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
            let vt = match ty {
                Some(VarType::Str(w)) => VarType::Str(w),
                Some(VarType::StrL) => VarType::StrL,
                _ => util::str_type_for(maxlen),
            };
            var = Variable::new(name, vt);
            var.format = format!("%{}s", match vt { VarType::Str(w) => w, _ => 9 });
            var.data = strs
                .into_iter()
                .map(|s| {
                    if s.is_empty() {
                        nmiss += 1;
                    }
                    let s = match vt {
                        VarType::Str(w) if s.len() > w => truncate_bytes(&s, w),
                        _ => s,
                    };
                    Value::Str(s)
                })
                .collect();
        } else {
            if matches!(ty, Some(VarType::Str(_)) | Some(VarType::StrL)) {
                return Err(rc_err(109, "type mismatch"));
            }
            let nums: Vec<f64> = vals
                .iter()
                .map(|v| match v {
                    Some(Val::Num(x)) => Ok(*x),
                    Some(Val::Str(_)) => Err(rc_err(109, "type mismatch")),
                    None => Ok(SYSMISS),
                })
                .collect::<Result<_, _>>()?;
            let vt = ty.unwrap_or(st.default_type);
            var = Variable::new(name, vt);
            var.data = nums
                .into_iter()
                .map(|x| {
                    if is_missing(x) {
                        nmiss += 1;
                    }
                    store_num(vt, x)
                })
                .collect();
        }
        if let Some(l) = lblname {
            var.value_label_name = l;
        }
        ds.add_variable(var)?;
        place_new_var(ds, name, cmd)?;
        st.data_changed = true;
        Ok(if nmiss > 0 {
            format!("({} generated)", plural(nmiss, "missing value"))
        } else {
            String::new()
        })
    }
}

fn truncate_bytes(s: &str, w: usize) -> String {
    let mut end = w.min(s.len());
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

/// Store a number into a storage type, emulating precision loss.
pub fn store_num(vt: VarType, x: f64) -> Value {
    if is_missing(x) {
        return Value::Double(ferrostat_core::missing::normalize(x));
    }
    match vt {
        VarType::Byte => {
            if fits(VarType::Byte, x) { Value::Byte(x as i8) } else { Value::Double(SYSMISS) }
        }
        VarType::Int => {
            if fits(VarType::Int, x) { Value::Int(x as i16) } else { Value::Double(SYSMISS) }
        }
        VarType::Long => {
            if fits(VarType::Long, x) { Value::Long(x as i32) } else { Value::Double(SYSMISS) }
        }
        VarType::Float => {
            if x.abs() > 1.70141173319e38 { Value::Double(SYSMISS) } else { Value::Float(x as f32) }
        }
        _ => Value::Double(x),
    }
}

/// Handle before()/after() placement of a newly created variable.
pub fn place_new_var(ds: &mut DataSet, name: &str, cmd: &Command) -> Result<(), String> {
    if let Some(b) = cmd.opt("before", 3) {
        let target = varlist::resolve_name(ds, b.trim())?;
        move_vars(ds, &[name.to_string()], &target, true);
    } else if let Some(a) = cmd.opt("after", 3) {
        let target = varlist::resolve_name(ds, a.trim())?;
        move_vars(ds, &[name.to_string()], &target, false);
    }
    Ok(())
}

pub fn parse_type(t: &str) -> Result<VarType, String> {
    VarType::from_type_str(t).ok_or_else(|| rc_err(198, format!("{t} invalid type")))
}

/// Does the expression read the target variable at other observations
/// (`x[_n-1]`, `L.x`)? Then replace must run observation by observation.
fn refers_back(e: &ferrostat_parser::Expr, target: &str) -> bool {
    use ferrostat_parser::Expr::*;
    match e {
        Index(b, i) => matches!(&**b, Var(n) if n == target) || refers_back(b, target) || refers_back(i, target),
        Var(n) => n.split_once('.').map_or(false, |(op, v)| v == target && op.chars().all(|c| "lfdsLFDS0123456789".contains(c))),
        Binary(a, _, b) => refers_back(a, target) || refers_back(b, target),
        Unary(_, a) | Paren(a) => refers_back(a, target),
        Call(_, args) => args.iter().any(|a| refers_back(a, target)),
        _ => false,
    }
}

/// Sequential replace: each observation sees the values already replaced.
fn replace_sequential(ds: &mut DataSet, st: &mut Store, name: &str, e: &ferrostat_parser::Expr, mask: &[bool]) -> Result<String, String> {
    let vi = ds.var_index[name];
    if ds.vars[vi].var_type != VarType::Double {
        let old = ds.vars[vi].var_type;
        // promote so intermediate results are not truncated
        let nt = promote(old, VarType::Double);
        ds.vars[vi].data = ds.vars[vi].data.iter().map(|v| typed_value(nt, v.num())).collect();
        ds.vars[vi].var_type = nt;
    }
    let groups = st.by_groups.clone();
    let mut changes = 0;
    for i in 0..ds.n_obs {
        if !mask[i] {
            continue;
        }
        let v = {
            let env = Env::with_store(ds, st).groups(groups.as_ref());
            env.eval(e, i)?.num()?
        };
        let old = ds.vars[vi].data[i].num();
        if old != v {
            changes += 1;
        }
        ds.vars[vi].data[i] = store_num(ds.vars[vi].var_type, v);
    }
    st.data_changed = true;
    Ok(format!("({} made)", plural(changes, "real change")))
}

pub struct Replace;

impl CommandHandler for Replace {
    fn name(&self) -> &'static str {
        "replace"
    }

    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let (lhs, rhs) = split_assignment(cmd)?;
        let name = varlist::resolve_name(ds, lhs.trim())?;
        let mask = util::touse(ds, st, cmd)?;
        let parsed = parse_expression(&rhs).map_err(|e| rc_err(198, e.message))?;
        if refers_back(&parsed, &name) && !ds.vars[ds.var_index[&name]].var_type.is_string() {
            return replace_sequential(ds, st, &name, &parsed, &mask);
        }
        let vals = eval_over(ds, st, &rhs, &mask)?;
        let vi = ds.var_index[&name];
        let is_str = ds.vars[vi].var_type.is_string();
        let mut changes = 0;
        if is_str {
            let mut maxlen = 0;
            let mut newvals = Vec::new();
            for (i, v) in vals.into_iter().enumerate() {
                if let Some(v) = v {
                    let s = match v {
                        Val::Str(s) => s,
                        Val::Num(_) => return Err(rc_err(109, "type mismatch")),
                    };
                    maxlen = maxlen.max(s.len());
                    newvals.push((i, s));
                }
            }
            let var = &mut ds.vars[vi];
            if let VarType::Str(w) = var.var_type {
                if maxlen > w {
                    var.var_type = util::str_type_for(maxlen);
                    var.format = format!("%{}s", maxlen.min(2045));
                }
            }
            for (i, s) in newvals {
                if var.data[i].as_string() != Some(s.as_str()) {
                    changes += 1;
                }
                var.data[i] = Value::Str(s);
            }
        } else {
            let mut newvals = Vec::new();
            for (i, v) in vals.into_iter().enumerate() {
                if let Some(v) = v {
                    let x = match v {
                        Val::Num(x) => x,
                        Val::Str(_) => return Err(rc_err(109, "type mismatch")),
                    };
                    newvals.push((i, x));
                }
            }
            let var = &mut ds.vars[vi];
            let need = best_numeric_type(&newvals.iter().map(|x| x.1).collect::<Vec<_>>());
            let mut promoted = None;
            if newvals.iter().any(|&(_, x)| !fits(var.var_type, x)) {
                let nt = promote(var.var_type, need);
                if nt != var.var_type {
                    promoted = Some(nt);
                    var.var_type = nt;
                }
            }
            for (i, x) in newvals {
                let old = var.data[i].num();
                let new = store_num(var.var_type, x);
                if old != new.num() && !(is_missing(old) && is_missing(new.num()) && old == new.num()) {
                    changes += 1;
                }
                var.data[i] = new;
            }
            if let Some(nt) = promoted {
                st.data_changed = true;
                return Ok(format!("variable {name} was {} now {}\n({} made)", "promoted", nt.type_str(), plural(changes, "real change")));
            }
        }
        st.data_changed = true;
        Ok(format!("({} made)", plural(changes, "real change")))
    }
}

// ---- drop / keep ----------------------------------------------------------

pub struct Drop;

impl CommandHandler for Drop {
    fn name(&self) -> &'static str {
        "drop"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        drop_keep(ds, st, cmd, true)
    }
}

pub struct Keep;

impl CommandHandler for Keep {
    fn name(&self) -> &'static str {
        "keep"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        drop_keep(ds, st, cmd, false)
    }
}

fn drop_keep(ds: &mut DataSet, st: &mut Store, cmd: &Command, drop: bool) -> Result<String, String> {
    st.data_changed = true;
    if cmd.if_expr.is_some() || cmd.in_range.is_some() {
        if !cmd.main.trim().is_empty() {
            return Err(rc_err(198, "invalid syntax"));
        }
        let sel = util::touse(ds, st, cmd)?;
        let mask: Vec<bool> = sel.iter().map(|&s| s != drop).collect();
        let before = ds.n_obs;
        ds.keep_if(&mask)?;
        let gone = before - ds.n_obs;
        return Ok(format!("({} deleted)", plural(gone, "observation")));
    }
    if cmd.main.trim().is_empty() {
        return Err(rc_err(100, "varlist required"));
    }
    if drop && cmd.main.trim() == "_all" {
        *ds = DataSet::new();
        return Ok(String::new());
    }
    let vars = varlist::expand_unique(ds, &cmd.main)?;
    if drop {
        for v in &vars {
            ds.drop_variable(v)?;
        }
    } else {
        ds.keep_variables(&vars);
        // keep listed order of existing dataset (Stata keeps dataset order)
    }
    Ok(String::new())
}

// ---- rename / order -------------------------------------------------------

pub struct Rename;

impl CommandHandler for Rename {
    fn name(&self) -> &'static str {
        "rename"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let text = cmd.main.trim();
        let case = ["lower", "upper", "proper"].into_iter().find(|c| cmd.has_opt(c, 3));
        if let Some(case) = case {
            let vars = if text.is_empty() || text == "*" || text == "_all" { ds.varlist() } else { varlist::expand(ds, text.trim_start_matches('(').trim_end_matches(')'))? };
            let wildcard = text.is_empty() || text == "*" || text == "_all";
            for v in vars {
                // system-style names such as _ID are left alone by `rename *, lower`
                // (observed in Stata's log for test-andrea/Spatial.do)
                if wildcard && v.starts_with('_') {
                    continue;
                }
                let new = match case {
                    "lower" => v.to_lowercase(),
                    "upper" => v.to_uppercase(),
                    _ => {
                        let mut c = v.chars();
                        match c.next() {
                            Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
                            None => v.clone(),
                        }
                    }
                };
                if new != v {
                    rename_one(ds, &v, &new)?;
                }
            }
            st.data_changed = true;
            return Ok(String::new());
        }
        let (old, new): (Vec<String>, Vec<String>) = if text.starts_with('(') {
            let close = text.find(')').ok_or_else(|| rc_err(198, "invalid syntax"))?;
            let a = &text[1..close];
            let rest = text[close + 1..].trim();
            let b = rest.trim_start_matches('(').trim_end_matches(')');
            (varlist::expand(ds, a)?, b.split_whitespace().map(String::from).collect())
        } else {
            let w: Vec<&str> = text.split_whitespace().collect();
            if w.len() != 2 {
                return Err(rc_err(198, "invalid syntax"));
            }
            if w[0].contains(['*', '?']) {
                return rename_wild(ds, w[0], w[1]);
            }
            (vec![varlist::resolve_name(ds, w[0])?], vec![w[1].to_string()])
        };
        if old.len() != new.len() {
            return Err(rc_err(198, "number of old and new names differ"));
        }
        if cmd.has_opt("lower", 3) || cmd.has_opt("upper", 3) {
            // handled by rename_case below
        }
        for (o, n) in old.iter().zip(&new) {
            rename_one(ds, o, n)?;
        }
        st.data_changed = true;
        Ok(String::new())
    }
}

fn rename_wild(ds: &mut DataSet, from: &str, to: &str) -> Result<String, String> {
    // supports one '*' in from and to: prefix/suffix changes
    let (fp, fs) = from.split_once('*').ok_or_else(|| rc_err(198, "invalid syntax"))?;
    let (tp, ts) = to.split_once('*').unwrap_or((to, ""));
    let names = ds.varlist();
    let mut n = 0;
    for name in names {
        if name.starts_with(fp) && name.ends_with(fs) && name.len() >= fp.len() + fs.len() {
            let mid = &name[fp.len()..name.len() - fs.len()];
            rename_one(ds, &name, &format!("{tp}{mid}{ts}"))?;
            n += 1;
        }
    }
    if n == 0 {
        return Err(rc_err(111, format!("variable {from} not found")));
    }
    Ok(String::new())
}

pub fn rename_one(ds: &mut DataSet, old: &str, new: &str) -> Result<(), String> {
    varlist::check_name(new)?;
    if old == new {
        return Ok(());
    }
    if ds.var_index.contains_key(new) {
        return Err(rc_err(110, format!("variable {new} already defined")));
    }
    let i = ds.var_index.remove(old).ok_or_else(|| rc_err(111, format!("variable {old} not found")))?;
    ds.vars[i].name = new.to_string();
    ds.var_index.insert(new.to_string(), i);
    for s in ds.sort_by.iter_mut() {
        if s == old {
            *s = new.to_string();
        }
    }
    if let Some(c) = ds.characteristics.remove(old) {
        ds.characteristics.insert(new.to_string(), c);
    }
    Ok(())
}

fn rebuild_index(ds: &mut DataSet) {
    ds.var_index.clear();
    for (i, v) in ds.vars.iter().enumerate() {
        ds.var_index.insert(v.name.clone(), i);
    }
}

/// Move `vars` before/after `target`.
pub fn move_vars(ds: &mut DataSet, vars: &[String], target: &str, before: bool) {
    let mut moved = Vec::new();
    for v in vars {
        if let Some(pos) = ds.vars.iter().position(|x| &x.name == v) {
            moved.push(ds.vars.remove(pos));
        }
    }
    let tpos = ds.vars.iter().position(|x| x.name == target).unwrap_or(ds.vars.len());
    let at = if before { tpos } else { (tpos + 1).min(ds.vars.len()) };
    for (k, v) in moved.into_iter().enumerate() {
        ds.vars.insert(at + k, v);
    }
    rebuild_index(ds);
}

pub struct Order;

impl CommandHandler for Order {
    fn name(&self) -> &'static str {
        "order"
    }
    fn execute(&self, ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand_unique(ds, &cmd.main)?;
        if cmd.has_opt("alphabetic", 5) || cmd.has_opt("sequential", 3) {
            let mut sorted = vars.clone();
            sorted.sort();
            return order_to(ds, &sorted, cmd);
        }
        order_to(ds, &vars, cmd)
    }
}

fn order_to(ds: &mut DataSet, vars: &[String], cmd: &Command) -> Result<String, String> {
    if let Some(b) = cmd.opt("before", 3) {
        let t = varlist::resolve_name(ds, b.trim())?;
        move_vars(ds, vars, &t, true);
    } else if let Some(a) = cmd.opt("after", 3) {
        let t = varlist::resolve_name(ds, a.trim())?;
        move_vars(ds, vars, &t, false);
    } else if cmd.has_opt("last", 4) {
        let mut moved = Vec::new();
        for v in vars {
            if let Some(pos) = ds.vars.iter().position(|x| &x.name == v) {
                moved.push(ds.vars.remove(pos));
            }
        }
        ds.vars.extend(moved);
        rebuild_index(ds);
    } else {
        let mut moved = Vec::new();
        for v in vars {
            if let Some(pos) = ds.vars.iter().position(|x| &x.name == v) {
                moved.push(ds.vars.remove(pos));
            }
        }
        for (k, v) in moved.into_iter().enumerate() {
            ds.vars.insert(k, v);
        }
        rebuild_index(ds);
    }
    Ok(String::new())
}

/// `move var1 var2`: put var1 before var2.
pub struct Move;

impl CommandHandler for Move {
    fn name(&self) -> &'static str {
        "move"
    }
    fn execute(&self, ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let v = varlist::expand(ds, &cmd.main)?;
        if v.len() != 2 {
            return Err(rc_err(198, "invalid syntax"));
        }
        move_vars(ds, &v[..1], &v[1], true);
        Ok(String::new())
    }
}

// ---- sorting --------------------------------------------------------------

pub struct Sort;

impl CommandHandler for Sort {
    fn name(&self) -> &'static str {
        "sort"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        if vars.is_empty() {
            return Err(rc_err(100, "varlist required"));
        }
        if let Some(r) = &cmd.in_range {
            let (a, b) = r.resolve(ds.n_obs).map_err(|e| rc_err(198, e))?;
            sort_range(ds, &vars, &vec![false; vars.len()], a, b);
            ds.sort_by.clear();
        } else {
            ds.sort(&vars, true)?;
        }
        st.data_changed = true;
        Ok(String::new())
    }
}

/// Stable sort of observations [a, b) by keys; `desc[k]` reverses key k.
pub fn sort_range(ds: &mut DataSet, vars: &[String], desc: &[bool], a: usize, b: usize) {
    sort_range_mfirst(ds, vars, desc, a, b, false)
}

fn sort_range_mfirst(ds: &mut DataSet, vars: &[String], desc: &[bool], a: usize, b: usize, mfirst: bool) {
    let cols: Vec<usize> = vars.iter().filter_map(|v| ds.var_index.get(v).copied()).collect();
    let mut idx: Vec<usize> = (a..b).collect();
    idx.sort_by(|&x, &y| {
        for (k, &c) in cols.iter().enumerate() {
            let (vx, vy) = (&ds.vars[c].data[x], &ds.vars[c].data[y]);
            let mut o = vx.partial_cmp(vy).unwrap_or(std::cmp::Ordering::Equal);
            if desc[k] {
                o = o.reverse();
                if mfirst {
                    // keep missing first when descending with mfirst
                } else if vx.is_missing() != vy.is_missing() {
                    // Stata's gsort -x puts missing last by default
                    o = if vx.is_missing() { std::cmp::Ordering::Greater } else { std::cmp::Ordering::Less };
                }
            }
            if o != std::cmp::Ordering::Equal {
                return o;
            }
        }
        std::cmp::Ordering::Equal
    });
    for var in ds.vars.iter_mut() {
        let tmp: Vec<Value> = idx.iter().map(|&i| var.data[i].clone()).collect();
        for (k, v) in tmp.into_iter().enumerate() {
            var.data[a + k] = v;
        }
    }
}

pub struct Gsort;

impl CommandHandler for Gsort {
    fn name(&self) -> &'static str {
        "gsort"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut vars = Vec::new();
        let mut desc = Vec::new();
        let mut pending: Option<bool> = None;
        for tok in cmd.main.split_whitespace() {
            if tok == "-" || tok == "+" {
                pending = Some(tok == "-");
                continue;
            }
            let (d, name) = if let Some(r) = tok.strip_prefix('-') {
                (true, r)
            } else if let Some(r) = tok.strip_prefix('+') {
                (false, r)
            } else {
                (pending.unwrap_or(false), tok)
            };
            pending = None;
            for v in varlist::expand(ds, name)? {
                vars.push(v);
                desc.push(d);
            }
        }
        if vars.is_empty() {
            return Err(rc_err(100, "varlist required"));
        }
        let n = ds.n_obs;
        sort_range_mfirst(ds, &vars, &desc, 0, n, cmd.has_opt("mfirst", 2));
        ds.sort_by = if desc.iter().any(|&d| d) { Vec::new() } else { vars.clone() };
        if let Some(g) = cmd.opt("generate", 3) {
            let name = g.trim();
            let groups = util::groups_by(ds, &vars);
            let mut id = 0.0;
            let mut vals = vec![0.0; n];
            for i in 0..n {
                if groups.start[i] == i {
                    id += 1.0;
                }
                vals[i] = id;
            }
            ds.add_variable(util::new_double(name, vals))?;
        }
        st.data_changed = true;
        Ok(String::new())
    }
}

// ---- format / compress ----------------------------------------------------

pub struct Format;

impl CommandHandler for Format {
    fn name(&self) -> &'static str {
        "format"
    }
    fn execute(&self, ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let words: Vec<&str> = cmd.raw.split_whitespace().skip(1).collect();
        let (fmt, vars): (String, Vec<String>) = if let Some(pos) = words.iter().position(|w| w.starts_with('%')) {
            let f = words[pos].to_string();
            let rest: Vec<&str> = words.iter().enumerate().filter(|(i, _)| *i != pos).map(|(_, w)| *w).collect();
            (f, varlist::expand(ds, &rest.join(" "))?)
        } else {
            // `format varlist` lists formats
            let vars = varlist::expand(ds, &words.join(" "))?;
            let lines: Vec<String> = vars.iter().map(|v| format!("{:<14} {}", v, ds.get_var(v).unwrap().format)).collect();
            return Ok(lines.join("\n"));
        };
        if crate::format::NumFormat::parse(&fmt).is_none() && !fmt.starts_with("%t") {
            return Err(rc_err(120, format!("invalid %format {fmt}")));
        }
        for v in vars {
            let var = ds.get_var_mut(&v).unwrap();
            let is_s = fmt.ends_with('s');
            if var.var_type.is_string() != is_s {
                return Err(rc_err(120, format!("invalid %format for {v}")));
            }
            var.format = fmt.clone();
        }
        Ok(String::new())
    }
}

pub struct Compress;

impl CommandHandler for Compress {
    fn name(&self) -> &'static str {
        "compress"
    }
    fn execute(&self, ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = if cmd.main.trim().is_empty() { ds.varlist() } else { varlist::expand(ds, &cmd.main)? };
        let mut out = Vec::new();
        let mut saved = 0usize;
        for name in vars {
            let n = ds.n_obs;
            let var = ds.get_var_mut(&name).unwrap();
            let old = var.var_type;
            let new = match old {
                VarType::Str(w) => {
                    let m = var.data.iter().map(|v| v.as_string().map_or(0, |s| s.len())).max().unwrap_or(1).max(1);
                    if m < w { VarType::Str(m) } else { old }
                }
                VarType::StrL => old,
                _ => {
                    let vals: Vec<f64> = var.data.iter().map(|v| v.num()).collect();
                    let t = best_numeric_type(&vals);
                    let t = if t == VarType::Float && old == VarType::Double {
                        // only demote doubles to float when exact
                        if vals.iter().all(|&x| fits(VarType::Float, x)) { VarType::Float } else { VarType::Double }
                    } else {
                        t
                    };
                    if t.storage_width() < old.storage_width() { t } else { old }
                }
            };
            if new != old {
                saved += (old.storage_width() - new.storage_width()) * n;
                var.var_type = new;
                if new.is_numeric() {
                    var.data = var.data.iter().map(|v| typed_value(new, v.num())).collect();
                    if var.format == "%9.0g" && matches!(new, VarType::Byte | VarType::Int) {
                        var.format = "%8.0g".into();
                    }
                } else if let VarType::Str(w) = new {
                    var.format = format!("%{w}s");
                }
                out.push(format!("  variable {} was {} now {}", name, old.type_str(), new.type_str()));
            }
        }
        if out.is_empty() {
            Ok("  (0 bytes saved)".into())
        } else {
            out.push(format!("  ({} bytes saved)", saved));
            Ok(out.join("\n"))
        }
    }
}

// ---- assert / insobs ------------------------------------------------------

pub struct Assert;

impl CommandHandler for Assert {
    fn name(&self) -> &'static str {
        "assert"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mask = util::touse(ds, st, cmd)?;
        let e = parse_expression(&cmd.main).map_err(|e| rc_err(198, e.message))?;
        let env = Env::with_store(ds, st).groups(st.by_groups.as_ref());
        let mut bad = 0;
        let mut considered = 0;
        for i in 0..ds.n_obs {
            if mask[i] {
                considered += 1;
                if env.eval(&e, i)?.num()? == 0.0 {
                    bad += 1;
                }
            }
        }
        if ds.n_obs == 0 || considered == 0 {
            // assert on constants with no data
            if considered == 0 && ds.n_obs == 0 && env.eval(&e, 0).map(|v| v.is_true()).unwrap_or(true) {
                return Ok(String::new());
            }
            if cmd.has_opt("null", 4) {
                return Ok(String::new());
            }
            if ds.n_obs > 0 {
                return Err(rc_err(2000, "no observations"));
            }
        }
        if bad > 0 {
            return Err(rc_err(9, format!("{} contradiction{} in {} observation{}\nassertion is false", bad, if bad == 1 { "" } else { "s" }, considered, if considered == 1 { "" } else { "s" })));
        }
        Ok(String::new())
    }
}

pub struct Insobs;

impl CommandHandler for Insobs {
    fn name(&self) -> &'static str {
        "insobs"
    }
    fn execute(&self, ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let k: usize = cmd.main.trim().parse().map_err(|_| rc_err(198, "invalid number"))?;
        let at = if let Some(b) = cmd.opt("before", 1) {
            b.trim().parse::<usize>().map_err(|_| rc_err(198, "invalid before()"))? - 1
        } else if let Some(a) = cmd.opt("after", 1) {
            a.trim().parse::<usize>().map_err(|_| rc_err(198, "invalid after()"))?
        } else {
            ds.n_obs
        };
        for var in ds.vars.iter_mut() {
            let fill = if var.var_type.is_string() { Value::Str(String::new()) } else { Value::Double(SYSMISS) };
            for _ in 0..k {
                var.data.insert(at, fill.clone());
            }
        }
        ds.n_obs += k;
        Ok(String::new())
    }
}

/// `set obs N`
pub fn set_obs(ds: &mut DataSet, n: usize) -> Result<String, String> {
    if n < ds.n_obs {
        return Err(rc_err(198, format!("observation number out of range; current N is {}", ds.n_obs)));
    }
    while ds.n_obs < n {
        ds.add_observation();
    }
    Ok(format!("Number of observations (_N) was {}, now {}.", ds.n_obs.min(n), n))
}

/// Convert all values of a variable to a new numeric storage type.
pub fn recast(ds: &mut DataSet, name: &str, ty: VarType, force: bool) -> Result<(), String> {
    let var = ds.get_var_mut(name).ok_or_else(|| rc_err(111, format!("variable {name} not found")))?;
    if var.var_type.is_string() != ty.is_string() {
        return Err(rc_err(109, "type mismatch"));
    }
    if ty.is_numeric() {
        if !force && var.data.iter().any(|v| !fits(ty, v.num())) {
            return Err(rc_err(109, format!("{name}: {} values would be changed; not changed", ty.type_str())));
        }
        var.data = var.data.iter().map(|v| store_num(ty, v.num())).collect();
    }
    var.var_type = ty;
    Ok(())
}

pub struct Recast;

impl CommandHandler for Recast {
    fn name(&self) -> &'static str {
        "recast"
    }
    fn execute(&self, ds: &mut DataSet, _st: &mut Store, cmd: &Command) -> Result<String, String> {
        let mut w = cmd.main.split_whitespace();
        let ty = parse_type(w.next().ok_or_else(|| rc_err(198, "type required"))?)?;
        let vars = varlist::expand(ds, &w.collect::<Vec<_>>().join(" "))?;
        for v in vars {
            recast(ds, &v, ty, cmd.has_opt("force", 5))?;
        }
        Ok(String::new())
    }
}

/// `tsset`/`xtset`: declare panel and time variables.
pub struct Tsset;

impl CommandHandler for Tsset {
    fn name(&self) -> &'static str {
        "tsset"
    }
    fn execute(&self, ds: &mut DataSet, st: &mut Store, cmd: &Command) -> Result<String, String> {
        let vars = varlist::expand(ds, &cmd.main)?;
        let xt = cmd.verb == "xtset";
        let (panel, time) = match (vars.len(), xt) {
            (0, _) => {
                return match &st.tsset {
                    Some((p, t)) => Ok(format!("panel variable: {}\ntime variable: {}", p.clone().unwrap_or_default(), t)),
                    None => Err(rc_err(111, "time variable not set")),
                };
            }
            (1, true) => (Some(vars[0].clone()), None),
            (1, false) => (None, Some(vars[0].clone())),
            (_, _) => (Some(vars[0].clone()), Some(vars[1].clone())),
        };
        let mut keys = Vec::new();
        if let Some(p) = &panel {
            keys.push(p.clone());
        }
        if let Some(t) = &time {
            keys.push(t.clone());
        }
        ds.sort(&keys, true)?;
        let mut out = Vec::new();
        if let Some(p) = &panel {
            out.push(format!("Panel variable: {p}"));
            ds.characteristics.entry("_dta".into()).or_default().insert("iis".into(), p.clone());
        }
        if let Some(t) = &time {
            let tv = util::col(ds, t)?;
            let lo = tv.iter().cloned().filter(|x| !is_missing(*x)).fold(f64::INFINITY, f64::min);
            let hi = tv.iter().cloned().filter(|x| !is_missing(*x)).fold(f64::NEG_INFINITY, f64::max);
            out.push(format!(" Time variable: {t}, {} to {}", crate::format::fmt_g(lo, 9), crate::format::fmt_g(hi, 9)));
            ds.characteristics.entry("_dta".into()).or_default().insert("tis".into(), t.clone());
        }
        st.tsset = time.clone().map(|t| (panel.clone(), t)).or_else(|| panel.clone().map(|p| (Some(p), String::new())));
        if time.is_none() {
            st.tsset = None;
            if let Some(p) = panel {
                st.tsset = Some((Some(p), String::new()));
            }
        }
        Ok(out.join("\n"))
    }
}
