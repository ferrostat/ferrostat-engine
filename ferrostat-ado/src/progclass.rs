//! A pragmatic subset of the class system: `class name { declarations }`
//! (in do-files or `name.class` files found on the ado-path), member
//! programs `program .name`, instances `.a = .name.new`, member access and
//! assignment (`.a.x = exp`, `` `.a.x' ``), arrays, built-ins (`.classname`,
//! `.isa`, `.arrnels`, `.isofclass`, `.objkey`, `.copy`, `.ref`,
//! `.Declare`, `.arrpush`, `.arrpop`, `.arrdropall`, `.arrdropel`),
//! `classexit`, and `classutil drop|describe|dir|cdir|which`.

use std::collections::BTreeMap;

use ferrostat_cmd::eval::{rc_err, Env, Val};
use ferrostat_core::missing::SYSMISS;
use ferrostat_parser::parser::{find_top_level, parse_expression, split_top_level_words};

use crate::interpreter::{AdoInterpreter, Flow};
use crate::macros::{macro_number, unquote};
use crate::script::Stmt;

#[derive(Clone, Debug)]
pub enum CVal {
    Num(f64),
    Str(String),
    Arr(Vec<CVal>),
    Obj(Inst),
}

#[derive(Clone, Debug)]
pub struct Inst {
    pub class: String,
    /// Members in declaration order.
    pub members: Vec<(String, CVal)>,
}

impl Inst {
    fn get(&self, n: &str) -> Option<&CVal> {
        self.members.iter().find(|m| m.0 == n).map(|m| &m.1)
    }
    fn get_mut(&mut self, n: &str) -> Option<&mut CVal> {
        self.members.iter_mut().find(|m| m.0 == n).map(|m| &mut m.1)
    }
}

#[derive(Clone, Debug)]
struct Decl {
    kind: String,
    name: String,
    init: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct ClassDef {
    parent: Option<String>,
    decls: Vec<Decl>,
    classwide: Vec<(String, CVal)>,
}

#[derive(Default)]
pub struct ClassState {
    defs: BTreeMap<String, ClassDef>,
    objects: BTreeMap<String, CVal>,
    /// Objects whose member programs are running (absolute paths).
    this: Vec<Vec<Seg>>,
    /// The class whose member programs are being defined.
    pub cur_class: Option<String>,
    exit_val: Option<String>,
}

impl ClassState {
    /// `discard`: drop instances and class definitions.
    pub fn discard(&mut self) {
        self.objects.clear();
        self.defs.clear();
        self.cur_class = None;
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Seg {
    name: String,
    index: Option<usize>,
}

const BUILTINS: &[&str] = &["classname", "isa", "objtype", "arrnels", "isofclass", "objkey", "uname", "copy", "ref", "new", "Declare", "arrpush", "arrpop", "arrdropall", "arrdropel", "Global", "Local", "Super"];

fn eval(me: &AdoInterpreter, text: &str) -> Result<Val, String> {
    let e = parse_expression(text).map_err(|e| rc_err(198, format!("{text}: {}", e.message)))?;
    Env::with_store(&me.engine.dataset, &me.engine.store).eval(&e, 0)
}

fn to_cval(v: Val) -> CVal {
    match v {
        Val::Num(n) => CVal::Num(n),
        Val::Str(s) => CVal::Str(s),
    }
}

fn text_of(v: &CVal) -> String {
    match v {
        CVal::Num(n) => macro_number(*n),
        CVal::Str(s) => s.clone(),
        CVal::Arr(a) => a.iter().map(text_of).collect::<Vec<_>>().join(" "),
        CVal::Obj(o) => format!(".{}", o.class),
    }
}

/// Split `.a.b[2].c` into segments (indices evaluated).
fn parse_path(me: &AdoInterpreter, text: &str) -> Result<Vec<Seg>, String> {
    let t = text.trim().strip_prefix('.').unwrap_or(text.trim());
    let mut segs = Vec::new();
    let mut cur = String::new();
    let mut depth = 0;
    let mut parts = Vec::new();
    for c in t.chars() {
        match c {
            '[' => {
                depth += 1;
                cur.push(c);
            }
            ']' => {
                depth -= 1;
                cur.push(c);
            }
            '.' if depth == 0 => parts.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    parts.push(cur);
    for p in parts {
        if p.is_empty() {
            return Err(rc_err(198, format!(".{t} invalid name")));
        }
        let (name, index) = match p.find('[') {
            Some(i) => {
                let inner = p[i + 1..].strip_suffix(']').ok_or_else(|| rc_err(198, format!("{p}: ] expected")))?;
                let k = eval(me, inner)?.num()?;
                if !(k >= 1.0) {
                    return Err(rc_err(198, format!("{p}: invalid array index")));
                }
                (p[..i].to_string(), Some(k as usize))
            }
            None => (p.clone(), None),
        };
        segs.push(Seg { name, index });
    }
    Ok(segs)
}

fn path_text(p: &[Seg]) -> String {
    p.iter().map(|s| match s.index {
        Some(i) => format!(".{}[{i}]", s.name),
        None => format!(".{}", s.name),
    }).collect()
}

fn index_into(v: &CVal, idx: Option<usize>) -> Option<&CVal> {
    match idx {
        None => Some(v),
        Some(k) => match v {
            CVal::Arr(a) => a.get(k - 1),
            _ => None,
        },
    }
}

fn class_chain(st: &ClassState, class: &str) -> Vec<String> {
    let mut out = vec![class.to_string()];
    let mut c = class.to_string();
    while let Some(p) = st.defs.get(&c).and_then(|d| d.parent.clone()) {
        if out.contains(&p) {
            break;
        }
        out.push(p.clone());
        c = p;
    }
    out
}

fn classwide<'a>(st: &'a ClassState, class: &str, member: &str) -> Option<(String, &'a CVal)> {
    for c in class_chain(st, class) {
        if let Some(v) = st.defs.get(&c).and_then(|d| d.classwide.iter().find(|m| m.0 == member)) {
            return Some((c, &v.1));
        }
    }
    None
}

/// Value at an absolute path.
fn get<'a>(st: &'a ClassState, path: &[Seg]) -> Option<&'a CVal> {
    let first = path.first()?;
    let mut cur = index_into(st.objects.get(&first.name)?, first.index)?;
    for seg in &path[1..] {
        let CVal::Obj(o) = cur else { return None };
        let v = match o.get(&seg.name) {
            Some(v) => v,
            None => classwide(st, &o.class, &seg.name)?.1,
        };
        cur = index_into(v, seg.index)?;
    }
    Some(cur)
}

fn assign_indexed(slot: &mut CVal, idx: Option<usize>, val: CVal) -> Result<(), String> {
    match idx {
        None => {
            check_type(slot, &val)?;
            *slot = val;
        }
        Some(k) => {
            let CVal::Arr(a) = slot else { return Err(rc_err(198, "not an array")) };
            if a.len() < k {
                a.resize(k, CVal::Num(SYSMISS));
            }
            a[k - 1] = val;
        }
    }
    Ok(())
}

fn check_type(old: &CVal, new: &CVal) -> Result<(), String> {
    match (old, new) {
        (CVal::Num(_), CVal::Str(_)) | (CVal::Str(_), CVal::Num(_)) => Err(rc_err(109, "type mismatch")),
        _ => Ok(()),
    }
}

/// Assign at an absolute path.
fn set(st: &mut ClassState, path: &[Seg], val: CVal) -> Result<(), String> {
    let n = path.len();
    if n == 1 {
        let s = &path[0];
        return match s.index {
            None => {
                st.objects.insert(s.name.clone(), val);
                Ok(())
            }
            Some(_) => {
                let slot = st.objects.entry(s.name.clone()).or_insert(CVal::Arr(Vec::new()));
                assign_indexed(slot, s.index, val)
            }
        };
    }
    let last = &path[n - 1];
    // classwide member of the parent object?
    if let Some(CVal::Obj(o)) = get(st, &path[..n - 1]) {
        if o.get(&last.name).is_none() {
            let class = o.class.clone();
            let owner = classwide(st, &class, &last.name).map(|x| x.0).ok_or_else(|| rc_err(111, format!("{} not found", path_text(path))))?;
            let def = st.defs.get_mut(&owner).unwrap();
            let slot = &mut def.classwide.iter_mut().find(|m| m.0 == last.name).unwrap().1;
            return assign_indexed(slot, last.index, val);
        }
    }
    let first = &path[0];
    let mut cur = st.objects.get_mut(&first.name).ok_or_else(|| rc_err(111, format!(".{} not found", first.name)))?;
    if let Some(k) = first.index {
        cur = match cur {
            CVal::Arr(a) => a.get_mut(k - 1).ok_or_else(|| rc_err(111, "array element not found"))?,
            _ => return Err(rc_err(198, "not an array")),
        };
    }
    for seg in &path[1..n - 1] {
        let CVal::Obj(o) = cur else { return Err(rc_err(198, format!("{} is not an object", path_text(path)))) };
        cur = o.get_mut(&seg.name).ok_or_else(|| rc_err(111, format!("{} not found", path_text(path))))?;
        if let Some(k) = seg.index {
            cur = match cur {
                CVal::Arr(a) => a.get_mut(k - 1).ok_or_else(|| rc_err(111, "array element not found"))?,
                _ => return Err(rc_err(198, "not an array")),
            };
        }
    }
    let CVal::Obj(o) = cur else { return Err(rc_err(198, format!("{} is not an object", path_text(&path[..n - 1])))) };
    let slot = o.get_mut(&last.name).ok_or_else(|| rc_err(111, format!("{} not found", path_text(path))))?;
    assign_indexed(slot, last.index, val)
}

/// Make a path relative to the running member program absolute.
fn absolute(me: &AdoInterpreter, path: Vec<Seg>) -> Vec<Seg> {
    let st = &me.prog.class;
    let Some(this) = st.this.last() else { return path };
    let first = &path[0].name;
    if first == "Global" {
        return path[1..].to_vec();
    }
    let in_this = match get(st, this) {
        Some(CVal::Obj(o)) => {
            o.get(first).is_some() || classwide(st, &o.class, first).is_some() || find_program(me, &o.class, first).is_some() || BUILTINS.contains(&first.as_str())
        }
        Some(_) => BUILTINS.contains(&first.as_str()),
        None => false,
    };
    if in_this {
        let mut p = this.clone();
        p.extend(path);
        p
    } else {
        path
    }
}

fn find_program(me: &AdoInterpreter, class: &str, name: &str) -> Option<String> {
    class_chain(&me.prog.class, class).into_iter().map(|c| format!("{c}.{name}")).find(|k| me.engine.store.programs.contains_key(k))
}

/// Load `name.class` from the ado-path when the class is not yet defined.
fn ensure_class(me: &mut AdoInterpreter, name: &str) -> Result<(), String> {
    if me.prog.class.defs.contains_key(name) {
        return Ok(());
    }
    let found = crate::progtools::find_on_path(me, &format!("{name}.class"), None, true, false);
    let path = found.first().cloned().ok_or_else(|| rc_err(111, format!("class {name} not found")))?;
    let saved = me.prog.class.cur_class.take();
    let r = me.run_do(&path, "", true, false);
    me.prog.class.cur_class = saved;
    r?;
    if !me.prog.class.defs.contains_key(name) {
        return Err(rc_err(111, format!("class {name} not found in {path}")));
    }
    Ok(())
}

fn init_value(me: &mut AdoInterpreter, d: &Decl) -> Result<CVal, String> {
    Ok(match d.kind.as_str() {
        "double" => match &d.init {
            Some(t) => CVal::Num(eval(me, t)?.num()?),
            None => CVal::Num(SYSMISS),
        },
        "string" => match &d.init {
            Some(t) => match eval(me, t) {
                Ok(Val::Str(s)) => CVal::Str(s),
                Ok(Val::Num(_)) => return Err(rc_err(109, format!("{}: type mismatch", d.name))),
                Err(_) => CVal::Str(unquote(t)),
            },
            None => CVal::Str(String::new()),
        },
        "array" => match &d.init {
            Some(t) => array_literal(me, t)?,
            None => CVal::Arr(Vec::new()),
        },
        class => CVal::Obj(instantiate(me, class)?),
    })
}

fn instantiate(me: &mut AdoInterpreter, class: &str) -> Result<Inst, String> {
    ensure_class(me, class)?;
    let chain = class_chain(&me.prog.class, class);
    let mut inst = Inst { class: class.to_string(), members: Vec::new() };
    for c in chain.iter().rev() {
        let decls = me.prog.class.defs[c].decls.clone();
        for d in decls {
            let v = init_value(me, &d)?;
            match inst.get_mut(&d.name) {
                Some(slot) => *slot = v,
                None => inst.members.push((d.name.clone(), v)),
            }
        }
    }
    Ok(inst)
}

fn array_literal(me: &AdoInterpreter, text: &str) -> Result<CVal, String> {
    let t = text.trim();
    let inner = t.strip_prefix('{').and_then(|x| x.strip_suffix('}')).ok_or_else(|| rc_err(198, format!("{t}: array literal expected")))?;
    let mut out = Vec::new();
    let mut rest = inner.trim().to_string();
    while !rest.is_empty() {
        let (item, tail) = match find_top_level(&rest, ',') {
            Some(i) => (rest[..i].to_string(), rest[i + 1..].trim().to_string()),
            None => (rest.clone(), String::new()),
        };
        if !item.trim().is_empty() {
            out.push(to_cval(eval(me, item.trim())?));
        }
        rest = tail;
    }
    Ok(CVal::Arr(out))
}

/// Run member program `prog` (full key) with `this` = `obj`; returns its
/// output and the `classexit` value.
fn call_member(me: &mut AdoInterpreter, key: &str, obj: Vec<Seg>, args: &str) -> Result<(String, Option<String>), String> {
    me.prog.class.this.push(obj);
    let saved = me.prog.class.exit_val.take();
    let r = me.run_program(key, args);
    let val = std::mem::replace(&mut me.prog.class.exit_val, saved);
    me.prog.class.this.pop();
    r.map(|out| (out, val))
}

/// Evaluate the right-hand side `.path [args]` of an assignment or a reference.
fn path_value(me: &mut AdoInterpreter, text: &str) -> Result<(CVal, String), String> {
    let t = text.trim();
    let (ptxt, args) = split_path(t);
    let raw = parse_path(me, ptxt)?;
    let last = raw.last().unwrap().name.clone();
    if last == "new" && raw.len() == 2 {
        let class = raw[0].name.clone();
        return Ok((CVal::Obj(instantiate(me, &class)?), args.to_string()));
    }
    let path = absolute(me, raw);
    let n = path.len();
    if n >= 2 {
        let parent = &path[..n - 1];
        if let Some(v) = get(&me.prog.class, parent).cloned() {
            match last.as_str() {
                "copy" | "ref" => return Ok((v, String::new())),
                "classname" => {
                    if let CVal::Obj(o) = &v {
                        return Ok((CVal::Str(o.class.clone()), String::new()));
                    }
                }
                "isa" | "objtype" => {
                    let k = match &v {
                        CVal::Num(_) => "double",
                        CVal::Str(_) => "string",
                        CVal::Arr(_) => "array",
                        CVal::Obj(_) => "class",
                    };
                    return Ok((CVal::Str(k.into()), String::new()));
                }
                "arrnels" => {
                    if let CVal::Arr(a) = &v {
                        return Ok((CVal::Num(a.len() as f64), String::new()));
                    }
                }
                "isofclass" => {
                    let want = args.trim().trim_start_matches('.');
                    let yes = matches!(&v, CVal::Obj(o) if class_chain(&me.prog.class, &o.class).iter().any(|c| c == want));
                    return Ok((CVal::Num(yes as i32 as f64), String::new()));
                }
                "objkey" | "uname" => return Ok((CVal::Str(path_text(parent)), String::new())),
                _ => {}
            }
            if let CVal::Obj(o) = &v {
                if o.get(&last).is_none() && classwide(&me.prog.class, &o.class, &last).is_none() {
                    if let Some(key) = find_program(me, &o.class, &last) {
                        let (out, val) = call_member(me, &key, parent.to_vec(), args)?;
                        me.emit(&out);
                        return Ok((CVal::Str(val.unwrap_or_default()), String::new()));
                    }
                }
            }
        }
    }
    let v = get(&me.prog.class, &path).cloned().ok_or_else(|| rc_err(111, format!("{} not found", path_text(&path))))?;
    Ok((v, String::new()))
}

/// `.path rest` -> (path text, rest).
fn split_path(t: &str) -> (&str, &str) {
    let mut depth = 0;
    for (i, c) in t.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => depth -= 1,
            c if depth == 0 && (c.is_whitespace() || c == '=') => return (&t[..i], t[i..].trim_start()),
            _ => {}
        }
    }
    (t, "")
}

/// Replace `` `.path' `` references by their values.
pub fn expand_refs(me: &mut AdoInterpreter, text: &str) -> Result<String, String> {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find("`.") {
        out.push_str(&rest[..i]);
        let after = &rest[i + 1..];
        let Some(j) = after.find('\'') else {
            out.push_str(&rest[i..]);
            return Ok(out);
        };
        let inner = &after[..j];
        let v = match path_value(me, inner) {
            Ok((v, _)) => text_of(&v),
            Err(e) if crate::interpreter::rc_of(&e) == 111 => String::new(),
            Err(e) => return Err(e),
        };
        out.push_str(&v);
        rest = &after[j + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// A statement starting with `.`: assignment or member-program call.
pub fn statement(me: &mut AdoInterpreter, text: &str) -> Result<String, String> {
    let (ptxt, rest) = split_path(text.trim());
    if let Some(rhs) = rest.strip_prefix('=').filter(|r| !r.starts_with('=')) {
        let target = absolute(me, parse_path(me, ptxt)?);
        let rhs = rhs.trim();
        let is_path = rhs.starts_with('.') && rhs[1..].starts_with(|c: char| c.is_alphabetic() || c == '_') && {
            let name = &split_path(rhs).0[1..];
            // `.a` ... `.z` alone are extended missing values unless such an object exists
            !(name.len() == 1 && name.chars().all(|c| c.is_ascii_lowercase()) && !me.prog.class.objects.contains_key(name))
        };
        let val = if is_path {
            let (v, args) = path_value(me, rhs)?;
            if let CVal::Obj(o) = &v {
                let is_new = rhs.split_whitespace().next().map_or(false, |p| p.ends_with(".new"));
                set(&mut me.prog.class, &target, v.clone())?;
                if is_new {
                    if let Some(key) = find_program(me, &o.class, "new") {
                        let (out, _) = call_member(me, &key, target, &args)?;
                        return Ok(out);
                    }
                }
                return Ok(String::new());
            }
            v
        } else if rhs.starts_with('{') {
            array_literal(me, rhs)?
        } else {
            to_cval(eval(me, rhs)?)
        };
        set(&mut me.prog.class, &target, val)?;
        return Ok(String::new());
    }
    // `.obj.method args`
    let raw = parse_path(me, ptxt)?;
    let path = absolute(me, raw);
    let n = path.len();
    let last = path[n - 1].name.clone();
    let parent = path[..n - 1].to_vec();
    match last.as_str() {
        "Declare" => {
            let w = split_top_level_words(rest);
            let (kind, name) = match w.as_slice() {
                [k, n, ..] => (k.clone(), n.clone()),
                _ => return Err(rc_err(198, ".Declare type name [= init]")),
            };
            let init = rest.find('=').map(|i| rest[i + 1..].trim().to_string());
            let v = init_value(me, &Decl { kind, name: name.clone(), init })?;
            if parent.is_empty() {
                me.prog.class.objects.insert(name, v);
                return Ok(String::new());
            }
            let mut p = parent.clone();
            p.push(Seg { name: name.clone(), index: None });
            // add the member to the instance
            add_member(&mut me.prog.class, &parent, &name, v)?;
            let _ = p;
            return Ok(String::new());
        }
        "arrpush" | "arrpop" | "arrdropall" | "arrdropel" => {
            let item = if last == "arrpush" { Some(to_cval(eval(me, rest)?)) } else { None };
            let el = if last == "arrdropel" { Some(eval(me, rest)?.num()? as usize) } else { None };
            let mut arr = match get(&me.prog.class, &parent) {
                Some(CVal::Arr(a)) => a.clone(),
                _ => return Err(rc_err(198, format!("{} is not an array", path_text(&parent)))),
            };
            match last.as_str() {
                "arrpush" => arr.push(item.unwrap()),
                "arrpop" => {
                    arr.pop();
                }
                "arrdropall" => arr.clear(),
                _ => {
                    let k = el.unwrap();
                    if k >= 1 && k <= arr.len() {
                        arr[k - 1] = CVal::Num(SYSMISS);
                    }
                }
            }
            replace_value(&mut me.prog.class, &parent, CVal::Arr(arr))?;
            return Ok(String::new());
        }
        _ => {}
    }
    if let Some(CVal::Obj(o)) = get(&me.prog.class, &parent).cloned() {
        if let Some(key) = find_program(me, &o.class, &last) {
            let (out, _) = call_member(me, &key, parent, rest)?;
            return Ok(out);
        }
    }
    if n == 2 && last == "new" {
        // `.classname.new` on its own: create and discard
        instantiate(me, &path[0].name)?;
        return Ok(String::new());
    }
    Err(rc_err(199, format!("{} is unrecognized", path_text(&path))))
}

fn replace_value(st: &mut ClassState, path: &[Seg], v: CVal) -> Result<(), String> {
    // like `set`, but without the type check
    if path.len() == 1 && path[0].index.is_none() {
        st.objects.insert(path[0].name.clone(), v);
        return Ok(());
    }
    let old = get(st, path).cloned();
    if let Some(o) = &old {
        if std::mem::discriminant(o) != std::mem::discriminant(&v) {
            return Err(rc_err(109, "type mismatch"));
        }
    }
    set(st, path, v)
}

fn add_member(st: &mut ClassState, parent: &[Seg], name: &str, v: CVal) -> Result<(), String> {
    let mut obj = match get(st, parent) {
        Some(CVal::Obj(o)) => o.clone(),
        _ => return Err(rc_err(198, format!("{} is not an object", path_text(parent)))),
    };
    if obj.get(name).is_some() {
        return Err(rc_err(110, format!("{}.{name} already declared", path_text(parent))));
    }
    obj.members.push((name.to_string(), v));
    replace_value(st, parent, CVal::Obj(obj))
}

/// `class name { declarations }` in a do-file or class file.
pub fn class_block(me: &mut AdoInterpreter, header: &str, body: &[Stmt]) -> Result<Flow, String> {
    let h = header.trim();
    let (name, opts) = match find_top_level(h, ',') {
        Some(i) => (h[..i].trim(), h[i + 1..].trim()),
        None => (h, ""),
    };
    ferrostat_cmd::varlist::check_name(name)?;
    let parent = opts.find("inherit(").map(|i| opts[i + 8..].split(')').next().unwrap_or("").trim().trim_start_matches('.').to_string()).filter(|p| !p.is_empty());
    if let Some(p) = &parent {
        ensure_class(me, p)?;
    }
    let mut def = ClassDef { parent, ..Default::default() };
    let mut wide = false;
    let mut lines: Vec<String> = Vec::new();
    for s in body {
        lines.push(s.text.trim().to_string());
    }
    for line in lines {
        let l = line.trim();
        if l.is_empty() || l.starts_with('*') || l.starts_with("//") {
            continue;
        }
        let lower = l.to_lowercase();
        if lower.starts_with("classwide") && lower.ends_with(':') {
            wide = true;
            continue;
        }
        if lower.starts_with("instance") && lower.ends_with(':') {
            wide = false;
            continue;
        }
        let (decl, init) = match find_top_level(l, '=') {
            Some(i) => (l[..i].trim(), Some(l[i + 1..].trim().to_string())),
            None => (l, None),
        };
        let w: Vec<&str> = decl.split_whitespace().collect();
        let (kind, names) = match w.as_slice() {
            [k, rest @ ..] if !rest.is_empty() => (k.to_string(), rest.to_vec()),
            _ => return Err(rc_err(198, format!("class {name}: invalid declaration: {l}"))),
        };
        let kind = match kind.as_str() {
            "double" | "real" | "scalar" | "numeric" => "double".to_string(),
            "string" | "str" => "string".to_string(),
            "array" => "array".to_string(),
            other => other.trim_start_matches('.').to_string(),
        };
        for n in names {
            let n = n.trim_start_matches('.');
            ferrostat_cmd::varlist::check_name(n)?;
            let d = Decl { kind: kind.clone(), name: n.to_string(), init: init.clone() };
            if wide {
                let v = init_value(me, &d)?;
                def.classwide.push((n.to_string(), v));
            } else {
                def.decls.push(d);
            }
        }
    }
    me.prog.class.defs.insert(name.to_string(), def);
    me.prog.class.cur_class = Some(name.to_string());
    Ok(Flow::Next)
}

/// `}, inherit(parent)` after a class block.
pub fn inherit_line(me: &mut AdoInterpreter, text: &str) -> Result<String, String> {
    let p = text.find("inherit(").map(|i| text[i + 8..].split(')').next().unwrap_or("").trim().trim_start_matches('.').to_string()).unwrap_or_default();
    let cur = me.prog.class.cur_class.clone().ok_or_else(|| rc_err(198, "inherit() outside a class definition"))?;
    if p.is_empty() {
        return Err(rc_err(198, "inherit(classname) expected"));
    }
    ensure_class(me, &p)?;
    me.prog.class.defs.get_mut(&cur).unwrap().parent = Some(p);
    Ok(String::new())
}

/// `classexit [exp]`: leave a member program, returning a value.
pub fn classexit(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let t = r.trim();
    let v = if t.is_empty() {
        String::new()
    } else {
        match eval(me, t) {
            Ok(Val::Num(n)) => macro_number(n),
            Ok(Val::Str(s)) => s,
            Err(_) => unquote(t),
        }
    };
    me.prog.class.exit_val = Some(v);
    Ok("__EXIT__".into())
}

fn describe_value(st: &ClassState, name: &str, v: &CVal, indent: usize, out: &mut Vec<String>) {
    let pad = " ".repeat(indent);
    match v {
        CVal::Num(n) => out.push(format!("{pad}double   {name:<12} = {}", macro_number(*n))),
        CVal::Str(s) => out.push(format!("{pad}string   {name:<12} = \"{s}\"")),
        CVal::Arr(a) => out.push(format!("{pad}array    {name:<12} [{} elements]", a.len())),
        CVal::Obj(o) => {
            out.push(format!("{pad}.{:<7} {name}", o.class));
            for (k, m) in &o.members {
                describe_value(st, &format!(".{k}"), m, indent + 4, out);
            }
        }
    }
}

/// `classutil drop|describe|dir|cdir|which`
pub fn classutil(me: &mut AdoInterpreter, r: &str) -> Result<String, String> {
    let (sub, rest) = r.split_once(char::is_whitespace).map(|(a, b)| (a, b.trim())).unwrap_or((r.trim(), ""));
    let (body, _opts) = crate::progtools::split_opts(rest);
    match sub {
        "drop" => {
            for w in body.split_whitespace() {
                let n = w.trim_start_matches('.');
                if n == "_all" {
                    me.prog.class.objects.clear();
                } else if me.prog.class.objects.remove(n).is_none() {
                    return Err(rc_err(111, format!(".{n} not found")));
                }
            }
            Ok(String::new())
        }
        "describe" | "des" | "d" => {
            let t = body.trim();
            let mut out = Vec::new();
            if let Some(n) = t.strip_prefix('.') {
                let path = absolute(me, parse_path(me, &format!(".{n}"))?);
                let v = get(&me.prog.class, &path).cloned().ok_or_else(|| rc_err(111, format!(".{n} not found")))?;
                if let CVal::Obj(o) = &v {
                    out.push(format!("{} is an instance of class {}", path_text(&path), o.class));
                    for (k, m) in &o.members {
                        describe_value(&me.prog.class, &format!(".{k}"), m, 4, &mut out);
                    }
                    for c in class_chain(&me.prog.class, &o.class) {
                        for (k, m) in &me.prog.class.defs[&c].classwide {
                            describe_value(&me.prog.class, &format!(".{k}"), m, 4, &mut out);
                        }
                    }
                } else {
                    describe_value(&me.prog.class, &path_text(&path), &v, 0, &mut out);
                }
                me.engine.store.begin_r().set_str("type", match &v { CVal::Num(_) => "double", CVal::Str(_) => "string", CVal::Arr(_) => "array", CVal::Obj(_) => "classtype" });
            } else {
                ensure_class(me, t)?;
                let def = me.prog.class.defs[t].clone();
                out.push(format!("class {t}{}", def.parent.as_ref().map(|p| format!(" (inherits from {p})")).unwrap_or_default()));
                out.push("  instance-specific:".into());
                for d in &def.decls {
                    out.push(format!("    {:<8} .{}{}", d.kind, d.name, d.init.as_ref().map(|i| format!(" = {i}")).unwrap_or_default()));
                }
                if !def.classwide.is_empty() {
                    out.push("  classwide:".into());
                    for (k, m) in &def.classwide {
                        describe_value(&me.prog.class, &format!(".{k}"), m, 4, &mut out);
                    }
                }
                let progs: Vec<String> = me.engine.store.programs.keys().filter_map(|k| k.strip_prefix(&format!("{t}.")).map(|p| format!(".{p}"))).collect();
                if !progs.is_empty() {
                    out.push(format!("  member programs: {}", progs.join(" ")));
                }
            }
            Ok(out.join("\n"))
        }
        "dir" | "" => {
            let pat = body.trim().trim_start_matches('.');
            let rows: Vec<String> = me
                .prog
                .class
                .objects
                .iter()
                .filter(|(k, _)| pat.is_empty() || crate::macros::glob_match(pat, k))
                .map(|(k, v)| {
                    let kind = match v {
                        CVal::Num(_) => "double".to_string(),
                        CVal::Str(_) => "string".to_string(),
                        CVal::Arr(a) => format!("array[{}]", a.len()),
                        CVal::Obj(o) => format!(".{}", o.class),
                    };
                    format!("  .{k:<20} {kind}")
                })
                .collect();
            Ok(rows.join("\n"))
        }
        "cdir" => {
            let pat = if body.trim().is_empty() { "*".to_string() } else { body.trim().to_string() };
            let mut names: Vec<String> = me.prog.class.defs.keys().filter(|k| crate::macros::glob_match(&pat, k)).cloned().collect();
            if let Ok(list) = ferrostat_core::vfs::list(".") {
                for (f, d) in list {
                    if let Some(n) = f.strip_suffix(".class") {
                        if !d && crate::macros::glob_match(&pat, n) && !names.iter().any(|x| x == n) {
                            names.push(n.to_string());
                        }
                    }
                }
            }
            names.sort();
            Ok(names.iter().map(|n| format!("  {n}")).collect::<Vec<_>>().join("\n"))
        }
        "which" => {
            let n = body.trim().trim_start_matches('.');
            let found = crate::progtools::find_on_path(me, &format!("{n}.class"), None, true, false);
            match found.first() {
                Some(p) => Ok(p.clone()),
                None if me.prog.class.defs.contains_key(n) => Ok(format!("class {n} (defined in memory)")),
                None => Err(rc_err(111, format!("class {n} not found"))),
            }
        }
        other => Err(rc_err(198, format!("classutil {other}: invalid subcommand"))),
    }
}
