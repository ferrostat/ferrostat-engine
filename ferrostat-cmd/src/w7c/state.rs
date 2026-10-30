//! Fitted models kept for postestimation, keyed by an id stored in e().
//! `estimates store/restore` keep the id, so restored results find their
//! model again.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::store::{Results, Store};

pub enum Fitted {
    Sem(Rc<super::sem::SemFit>),
    Gsem(Rc<super::gsem::GsemFit>),
    Fmm(Rc<super::fmm::FmmFit>),
}

thread_local! {
    static REG: RefCell<(u64, BTreeMap<u64, Rc<Fitted>>)> = RefCell::new((0, BTreeMap::new()));
}

pub fn put(e: &mut Results, f: Fitted) {
    let id = REG.with(|r| {
        let mut r = r.borrow_mut();
        r.0 += 1;
        let id = r.0;
        r.1.insert(id, Rc::new(f));
        // keep the registry bounded
        while r.1.len() > 64 {
            let k = *r.1.keys().next().unwrap();
            r.1.remove(&k);
        }
        id
    });
    e.set_str("_w7c_id", id.to_string());
}

pub fn get(st: &Store) -> Option<Rc<Fitted>> {
    let id: u64 = st.e.macros.get("_w7c_id")?.parse().ok()?;
    REG.with(|r| r.borrow().1.get(&id).cloned())
}

pub fn sem(st: &Store) -> Option<Rc<super::sem::SemFit>> {
    match get(st)?.as_ref() {
        Fitted::Sem(s) => Some(s.clone()),
        _ => None,
    }
}

pub fn gsem(st: &Store) -> Option<Rc<super::gsem::GsemFit>> {
    match get(st)?.as_ref() {
        Fitted::Gsem(s) => Some(s.clone()),
        _ => None,
    }
}

pub fn fmm(st: &Store) -> Option<Rc<super::fmm::FmmFit>> {
    match get(st)?.as_ref() {
        Fitted::Fmm(s) => Some(s.clone()),
        _ => None,
    }
}
