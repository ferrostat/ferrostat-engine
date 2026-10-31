//! File access used by every command. On native targets these are thin
//! wrappers over `std::fs`; on WebAssembly they operate on an in-memory
//! file system (files are uploaded/downloaded by the host page).

use std::io;

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use std::io::{self, Write};

    pub fn read(path: &str) -> io::Result<Vec<u8>> {
        if crate::net::is_url(path) {
            return crate::net::fetch_url(path);
        }
        std::fs::read(path)
    }
    pub fn write(path: &str, data: &[u8]) -> io::Result<()> {
        std::fs::write(path, data)
    }
    pub fn append(path: &str, data: &[u8]) -> io::Result<()> {
        std::fs::OpenOptions::new().create(true).append(true).open(path)?.write_all(data)
    }
    pub fn exists(path: &str) -> bool {
        std::path::Path::new(path).exists()
    }
    pub fn remove(path: &str) -> io::Result<()> {
        std::fs::remove_file(path)
    }
    pub fn remove_dir(path: &str) -> io::Result<()> {
        std::fs::remove_dir(path)
    }
    pub fn create_dir_all(path: &str) -> io::Result<()> {
        std::fs::create_dir_all(path)
    }
    pub fn copy(a: &str, b: &str) -> io::Result<()> {
        if crate::net::is_url(a) {
            return std::fs::write(b, crate::net::fetch_url(a)?);
        }
        std::fs::copy(a, b).map(|_| ())
    }
    /// (name, is_dir) entries of a directory.
    pub fn list(dir: &str) -> io::Result<Vec<(String, bool)>> {
        let mut v = Vec::new();
        for e in std::fs::read_dir(dir)?.flatten() {
            v.push((e.file_name().to_string_lossy().to_string(), e.file_type().map(|t| t.is_dir()).unwrap_or(false)));
        }
        Ok(v)
    }
    pub fn current_dir() -> String {
        std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_default()
    }
    pub fn set_current_dir(p: &str) -> io::Result<()> {
        std::env::set_current_dir(p)
    }
    pub fn temp_path(name: &str) -> String {
        std::env::temp_dir().join(name).display().to_string()
    }
    /// (is_dir, size in bytes, modification time in seconds since 1970).
    pub fn meta(path: &str) -> Option<(bool, u64, Option<u64>)> {
        let m = std::fs::metadata(path).ok()?;
        let t = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs());
        Some((m.is_dir(), m.len(), t))
    }
}

#[cfg(target_arch = "wasm32")]
mod imp {
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::io;

    thread_local! {
        static FILES: RefCell<BTreeMap<String, Vec<u8>>> = RefCell::new(BTreeMap::new());
        static DIRS: RefCell<std::collections::BTreeSet<String>> = RefCell::new(["/".to_string(), "/tmp".to_string()].into_iter().collect());
        static CWD: RefCell<String> = RefCell::new("/".to_string());
        /// paths written or removed since the last `take_changes` (true = removed)
        static CHANGES: RefCell<BTreeMap<String, bool>> = RefCell::new(BTreeMap::new());
    }

    fn mark(p: &str, removed: bool) {
        CHANGES.with(|c| c.borrow_mut().insert(p.to_string(), removed));
    }

    /// Files changed since the last call: (path, removed).
    pub fn take_changes() -> Vec<(String, bool)> {
        CHANGES.with(|c| std::mem::take(&mut *c.borrow_mut()).into_iter().collect())
    }

    /// Absolute, normalised path ("/a/b").
    pub fn norm(path: &str) -> String {
        let path = path.replace('\\', "/");
        let base = if path.starts_with('/') { String::new() } else { CWD.with(|c| c.borrow().clone()) };
        let mut parts: Vec<String> = Vec::new();
        for seg in format!("{base}/{path}").split('/') {
            match seg {
                "" | "." => {}
                ".." => {
                    parts.pop();
                }
                s => parts.push(s.to_string()),
            }
        }
        format!("/{}", parts.join("/"))
    }

    fn not_found(p: &str) -> io::Error {
        io::Error::new(io::ErrorKind::NotFound, format!("{p} not found"))
    }

    fn parent(p: &str) -> String {
        match p.rfind('/') {
            Some(0) | None => "/".into(),
            Some(i) => p[..i].to_string(),
        }
    }

    pub fn read(path: &str) -> io::Result<Vec<u8>> {
        if crate::net::is_url(path) {
            return crate::net::fetch_url(path);
        }
        let p = norm(path);
        FILES.with(|f| f.borrow().get(&p).cloned()).ok_or_else(|| not_found(&p))
    }
    pub fn write(path: &str, data: &[u8]) -> io::Result<()> {
        let p = norm(path);
        create_dir_all(&parent(&p))?;
        mark(&p, false);
        FILES.with(|f| f.borrow_mut().insert(p, data.to_vec()));
        Ok(())
    }
    pub fn append(path: &str, data: &[u8]) -> io::Result<()> {
        let p = norm(path);
        mark(&p, false);
        FILES.with(|f| f.borrow_mut().entry(p).or_default().extend_from_slice(data));
        Ok(())
    }
    pub fn exists(path: &str) -> bool {
        let p = norm(path);
        FILES.with(|f| f.borrow().contains_key(&p)) || DIRS.with(|d| d.borrow().contains(&p))
    }
    pub fn remove(path: &str) -> io::Result<()> {
        let p = norm(path);
        let r = FILES.with(|f| f.borrow_mut().remove(&p)).map(|_| ()).ok_or_else(|| not_found(&p));
        if r.is_ok() {
            mark(&p, true);
        }
        r
    }
    pub fn remove_dir(path: &str) -> io::Result<()> {
        let p = norm(path);
        DIRS.with(|d| d.borrow_mut().remove(&p));
        Ok(())
    }
    pub fn create_dir_all(path: &str) -> io::Result<()> {
        let p = norm(path);
        let mut acc = String::new();
        for seg in p.split('/').filter(|s| !s.is_empty()) {
            acc = format!("{acc}/{seg}");
            DIRS.with(|d| d.borrow_mut().insert(acc.clone()));
        }
        Ok(())
    }
    pub fn copy(a: &str, b: &str) -> io::Result<()> {
        let data = read(a)?;
        write(b, &data)
    }
    pub fn list(dir: &str) -> io::Result<Vec<(String, bool)>> {
        let d = norm(dir);
        let prefix = if d == "/" { "/".to_string() } else { format!("{d}/") };
        let mut out = Vec::new();
        FILES.with(|f| {
            for k in f.borrow().keys() {
                if let Some(rest) = k.strip_prefix(&prefix) {
                    if !rest.contains('/') {
                        out.push((rest.to_string(), false));
                    }
                }
            }
        });
        DIRS.with(|ds| {
            for k in ds.borrow().iter() {
                if let Some(rest) = k.strip_prefix(&prefix) {
                    if !rest.is_empty() && !rest.contains('/') {
                        out.push((rest.to_string(), true));
                    }
                }
            }
        });
        Ok(out)
    }
    pub fn current_dir() -> String {
        CWD.with(|c| c.borrow().clone())
    }
    pub fn set_current_dir(p: &str) -> io::Result<()> {
        let n = norm(p);
        if !DIRS.with(|d| d.borrow().contains(&n)) {
            return Err(not_found(&n));
        }
        CWD.with(|c| *c.borrow_mut() = n);
        Ok(())
    }
    pub fn temp_path(name: &str) -> String {
        format!("/tmp/{name}")
    }
    /// (is_dir, size in bytes, modification time); times are not tracked.
    pub fn meta(path: &str) -> Option<(bool, u64, Option<u64>)> {
        let p = norm(path);
        if let Some(n) = FILES.with(|f| f.borrow().get(&p).map(|v| v.len() as u64)) {
            return Some((false, n, None));
        }
        DIRS.with(|d| d.borrow().contains(&p)).then_some((true, 0, None))
    }

    /// All files in the in-memory file system (for the host page).
    pub fn all_files() -> Vec<(String, usize)> {
        FILES.with(|f| f.borrow().iter().map(|(k, v)| (k.clone(), v.len())).collect())
    }
}

pub use imp::*;

/// Download an http:// or https:// address (see [`crate::net`]).
pub use crate::net::fetch_url;

pub fn read_to_string(path: &str) -> io::Result<String> {
    let b = read(path)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}
