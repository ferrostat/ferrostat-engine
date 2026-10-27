//! Replications of the resampling prefixes (`bootstrap`, `simulate`,
//! `permute`, the BCa jackknife), run in parallel.
//!
//! Each replication draws from its own random stream, derived from one
//! number taken from the session generator and the replication number, so
//! results do not depend on how many replications run at once or in which
//! order they finish. Native builds use one thread per processor
//! (`set processors #` lowers it); each thread has its own interpreter with
//! a copy of the session state. The browser build hands the work to helper
//! workers when the page allows it (see `web_bridge`) and otherwise runs the
//! replications one after another, with the same streams.

use ferrostat_core::DataSet;

use crate::interpreter::AdoInterpreter;

/// How one replication builds its data from the base dataset.
#[derive(Clone)]
pub(crate) enum Draw {
    /// Resample sampling units with replacement within strata; `size`
    /// units per stratum (all by default); `idcluster` names the variable
    /// that gets the new cluster numbers.
    Bootstrap { units: Vec<Vec<usize>>, strata: Vec<Vec<usize>>, size: Option<usize>, idcluster: Option<String> },
    /// Shuffle `var` within strata (each unit is one observation).
    Permute { var: String, units: Vec<Vec<usize>>, strata: Vec<Vec<usize>> },
    /// Leave unit `rep` out (jackknife); no random numbers.
    LeaveOut { units: Vec<Vec<usize>> },
    /// Run the command on the data as it is (simulate).
    Unchanged,
}

/// Everything a replication needs besides the interpreter.
#[derive(Clone)]
pub(crate) struct Job {
    pub base: DataSet,
    /// `base` without observations (variables, labels, formats)
    pub shell: DataSet,
    pub command: String,
    pub draw: Draw,
    /// Evaluates the statistics after the command has run.
    pub stats: std::sync::Arc<dyn Fn(&AdoInterpreter) -> Vec<f64> + Send + Sync>,
    /// The statistics as (`b` | `se` | `exp`, coefficient name or expression),
    /// so another engine instance can evaluate them.
    pub spec: Vec<(String, String)>,
    pub seed: u64,
}

/// A well-mixed 64-bit seed for replication `rep`.
fn rep_seed(seed: u64, rep: usize) -> u64 {
    // splitmix64 of the pair
    let mut z = seed ^ (rep as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A fresh base seed from the session generator (advances it once).
pub(crate) fn base_seed(me: &AdoInterpreter) -> u64 {
    let mut rng = me.engine.store.rng.borrow_mut();
    let hi = (rng.uniform() * 4_294_967_296.0) as u64;
    let lo = (rng.uniform() * 4_294_967_296.0) as u64;
    (hi << 32) | lo
}

fn take_rows(job: &Job, rows: &[usize]) -> DataSet {
    let mut d = job.shell.clone();
    for (v, src) in d.vars.iter_mut().zip(&job.base.vars) {
        v.data = rows.iter().map(|&r| src.data[r].clone()).collect();
    }
    d.n_obs = rows.len();
    d
}

/// The data of replication `rep`, drawn with `uniform`.
fn data_for(job: &Job, rep: usize, uniform: &mut dyn FnMut() -> f64) -> DataSet {
    match &job.draw {
        Draw::Unchanged => job.base.clone(),
        Draw::LeaveOut { units } => {
            let rows: Vec<usize> = (0..units.len()).filter(|&k| k != rep).flat_map(|k| units[k].iter().copied()).collect();
            take_rows(job, &rows)
        }
        Draw::Bootstrap { units, strata, size, idcluster } => {
            let mut rows = Vec::new();
            let mut ids = Vec::new();
            let mut next = 0.0;
            for h in strata {
                let m = h.len();
                for _ in 0..size.unwrap_or(m) {
                    let j = ((uniform() * m as f64) as usize).min(m - 1);
                    next += 1.0;
                    for &r in &units[h[j]] {
                        rows.push(r);
                        ids.push(next);
                    }
                }
            }
            let mut d = take_rows(job, &rows);
            if let Some(id) = idcluster {
                if let Some(var) = d.get_var_mut(id) {
                    var.var_type = ferrostat_core::VarType::Long;
                    var.data = ids.iter().map(|&x| ferrostat_cmd::util::typed_value(ferrostat_core::VarType::Long, x)).collect();
                }
            }
            d
        }
        Draw::Permute { var, units, strata } => {
            let mut perm: Vec<usize> = (0..job.base.n_obs).collect();
            for h in strata {
                // Fisher-Yates shuffle of the rows of this stratum
                let rows: Vec<usize> = h.iter().map(|&k| units[k][0]).collect();
                let mut sh = rows.clone();
                for i in (1..sh.len()).rev() {
                    let j = ((uniform() * (i + 1) as f64) as usize).min(i);
                    sh.swap(i, j);
                }
                for (a, b) in rows.iter().zip(&sh) {
                    perm[*a] = *b;
                }
            }
            let mut d = job.base.clone();
            if let Some(col) = job.base.get_var(var).map(|v| v.data.clone()) {
                d.get_var_mut(var).unwrap().data = perm.iter().map(|&i| col[i].clone()).collect();
            }
            d
        }
    }
}

/// Run replication `rep` on `me`: `None` when the command failed.
pub(crate) fn run_one(me: &mut AdoInterpreter, job: &Job, rep: usize, noisily: bool) -> Option<Vec<f64>> {
    me.engine.store.rng.borrow_mut().reseed(rep_seed(job.seed, rep));
    let data = {
        let rng = &me.engine.store.rng;
        data_for(job, rep, &mut || rng.borrow_mut().uniform())
    };
    me.engine.dataset = data;
    match crate::resample::run_cmd(me, &job.command, noisily) {
        Ok(()) => Some((job.stats)(me)),
        Err(_) => None,
    }
}

/// Number of replications to run at once.
fn workers(me: &AdoInterpreter, reps: usize) -> usize {
    // Mata state lives only in the session's interpreter
    if !me.mata.is_empty() {
        return 1;
    }
    // the browser's helper workers decide how many run; `set processors`
    // still caps them
    #[cfg(target_arch = "wasm32")]
    let avail = if HOOK.get().is_some() { 256 } else { 1 };
    #[cfg(not(target_arch = "wasm32"))]
    let avail = std::thread::available_parallelism().map_or(1, |n| n.get());
    let set = me.engine.store.settings.get("processors").and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(avail);
    set.clamp(1, avail.max(1)).min(reps.max(1))
}

/// Run replications `from..reps` and return their results in order. The
/// session's generator is left as it was (`base_seed` advanced it).
pub(crate) fn run_all(me: &mut AdoInterpreter, job: &Job, from: usize, reps: usize, noisily: bool) -> Vec<Option<Vec<f64>>> {
    let count = reps.saturating_sub(from);
    let n = if noisily { 1 } else { workers(me, count) };
    let rng = me.engine.store.rng.borrow().clone();
    let out = if n <= 1 {
        (from..reps).map(|rep| run_one(me, job, rep, noisily)).collect()
    } else {
        parallel_reps(me, job, from, reps, n)
    };
    *me.engine.store.rng.borrow_mut() = rng;
    out
}

#[cfg(target_arch = "wasm32")]
fn parallel_reps(me: &mut AdoInterpreter, job: &Job, from: usize, reps: usize, _n: usize) -> Vec<Option<Vec<f64>>> {
    if let Some(hook) = HOOK.get() {
        if let Ok(data) = dta_bytes(&job.base) {
            if let Some(out) = hook(&job_text(me, job, from, reps), &data).and_then(|t| parse_results(&t, reps - from)) {
                return out;
            }
        }
    }
    (from..reps).map(|rep| run_one(me, job, rep, false)).collect()
}

/// Runs a job on other engine instances (the browser's helper workers):
/// `(job text, base data as .dta) -> JSON array with one array of
/// statistics or null per replication`, or `None` when unavailable.
pub type Hook = fn(&str, &[u8]) -> Option<String>;

/// Set by the browser build when helper workers can be used.
pub static HOOK: std::sync::OnceLock<Hook> = std::sync::OnceLock::new();

#[cfg(not(target_arch = "wasm32"))]
fn parallel_reps(me: &mut AdoInterpreter, job: &Job, from: usize, reps: usize, n: usize) -> Vec<Option<Vec<f64>>> {
    // `set parallel_transport text` sends the job the way the browser sends
    // it to helper workers (text and .dta bytes), to test that path
    if me.engine.store.settings.get("parallel_transport").is_some_and(|v| v == "text") {
        if let Ok(data) = dta_bytes(&job.base) {
            let text = job_text(me, job, from, reps);
            if let Some(out) = by_text(&text, &data, from, reps, n).and_then(|t| parse_results(&t, reps - from)) {
                return out;
            }
        }
    }
    threaded(me, job, from, reps, n)
}

/// Helper-worker protocol run on threads: each thread prepares the job from
/// its text and takes replications from a shared counter.
#[cfg(not(target_arch = "wasm32"))]
fn by_text(text: &str, data: &[u8], from: usize, reps: usize, n: usize) -> Option<String> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let next = AtomicUsize::new(from);
    let out: Mutex<Vec<Option<Vec<f64>>>> = Mutex::new(vec![None; reps - from]);
    let ok = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..n {
            let (next, out, ok) = (&next, &out, &ok);
            scope.spawn(move || {
                let Ok(mut p) = Prepared::new(text, data.to_vec()) else { return };
                ok.fetch_add(1, Ordering::Relaxed);
                loop {
                    let rep = next.fetch_add(1, Ordering::Relaxed);
                    if rep >= reps {
                        break;
                    }
                    let r = p.rep(rep);
                    out.lock().unwrap()[rep - from] = r;
                }
            });
        }
    });
    // no helper could prepare the job: let the caller run it
    (next.load(Ordering::Relaxed) >= reps).then(|| serde_json::to_string(&out.into_inner().unwrap()).unwrap())
}

// ---------------------------------------------------------------------------
// Jobs for other engine instances
// ---------------------------------------------------------------------------

fn dta_bytes(ds: &DataSet) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    ferrostat_core::dta::DtaWriter::write(&mut out, ds).map_err(|e| e.to_string())?;
    Ok(out)
}

fn num_text(x: f64) -> String {
    if ferrostat_core::missing::is_missing(x) { ".".into() } else { format!("{x:?}") }
}

/// Commands that recreate the session state a replication may use:
/// programs, globals, scalars, matrices and the time-series settings.
fn preamble(me: &AdoInterpreter) -> String {
    let st = &me.engine.store;
    let mut out = String::new();
    for p in st.programs.values() {
        let class = if p.class.is_empty() || p.class == "nclass" { String::new() } else { format!(", {}", p.class) };
        out.push_str(&format!("program define {}{class}\n{}\nend\n", p.name, p.body.join("\n")));
    }
    for (k, v) in &st.globals {
        out.push_str(&format!("global {k} `\"{v}\"'\n"));
    }
    for (k, v) in &st.scalars {
        out.push_str(&format!("scalar {k} = {}\n", num_text(*v)));
    }
    for (k, v) in &st.str_scalars {
        out.push_str(&format!("scalar {k} = `\"{v}\"'\n"));
    }
    for (k, m) in &st.matrices {
        let rows: Vec<String> = (0..m.rows).map(|i| (0..m.cols).map(|j| num_text(m.data[i * m.cols + j])).collect::<Vec<_>>().join(",")).collect();
        out.push_str(&format!("matrix {k} = ({})\n", rows.join("\\")));
        out.push_str(&format!("matrix rownames {k} = {}\nmatrix colnames {k} = {}\n", m.rownames.join(" "), m.colnames.join(" ")));
    }
    match &st.tsset {
        Some((Some(panel), time)) => out.push_str(&format!("quietly xtset {panel} {time}\n")),
        Some((None, time)) => out.push_str(&format!("quietly tsset {time}\n")),
        None => {}
    }
    out
}

/// Replications `from..reps` of `job` as text for another engine instance.
pub(crate) fn job_text(me: &AdoInterpreter, job: &Job, from: usize, reps: usize) -> String {
    let workers = workers(me, reps - from);
    use serde_json::json;
    let draw = match &job.draw {
        Draw::Bootstrap { units, strata, size, idcluster } => json!({"kind": "bootstrap", "units": units, "strata": strata, "size": size, "idcluster": idcluster}),
        Draw::Permute { var, units, strata } => json!({"kind": "permute", "var": var, "units": units, "strata": strata}),
        Draw::LeaveOut { units } => json!({"kind": "leaveout", "units": units}),
        Draw::Unchanged => json!({"kind": "unchanged"}),
    };
    json!({
        "command": job.command,
        "seed": job.seed.to_string(),
        "from": from,
        "reps": reps,
        "k": job.spec.len(),
        "workers": workers,
        "draw": draw,
        "stats": job.spec,
        "preamble": preamble(me),
    })
    .to_string()
}

fn parse_results(text: &str, n: usize) -> Option<Vec<Option<Vec<f64>>>> {
    let v: Vec<Option<Vec<f64>>> = serde_json::from_str(text).ok()?;
    (v.len() == n).then_some(v)
}

/// A job received from another engine instance, ready to run replications.
pub struct Prepared {
    it: AdoInterpreter,
    job: Job,
}

impl Prepared {
    /// Set up the session state and data described by `text` and `data`.
    pub fn new(text: &str, data: Vec<u8>) -> Result<Prepared, String> {
        use serde_json::Value;
        let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let base = ferrostat_core::dta::DtaReader::from_bytes(data).map_err(|e| e.to_string())?;
        let mut it = AdoInterpreter::new();
        it.engine.dataset = base.clone();
        it.quiet += 1;
        let pre = v["preamble"].as_str().unwrap_or("");
        if !pre.is_empty() {
            it.run_script(pre)?;
        }
        it.quiet -= 1;
        let units = |key: &str| -> Vec<Vec<usize>> { serde_json::from_value(v["draw"][key].clone()).unwrap_or_default() };
        let draw = match v["draw"]["kind"].as_str() {
            Some("bootstrap") => Draw::Bootstrap {
                units: units("units"),
                strata: units("strata"),
                size: v["draw"]["size"].as_u64().map(|x| x as usize),
                idcluster: v["draw"]["idcluster"].as_str().map(str::to_string),
            },
            Some("permute") => Draw::Permute { var: v["draw"]["var"].as_str().unwrap_or("").to_string(), units: units("units"), strata: units("strata") },
            Some("leaveout") => Draw::LeaveOut { units: units("units") },
            _ => Draw::Unchanged,
        };
        let spec: Vec<(String, String)> = serde_json::from_value(v["stats"].clone()).map_err(|e| e.to_string())?;
        let stats = crate::resample::stats_from_spec(&spec)?;
        let job = Job {
            shell: crate::resample::empty_shell(&base),
            base,
            command: v["command"].as_str().unwrap_or("").to_string(),
            draw,
            stats,
            spec,
            seed: v["seed"].as_str().and_then(|s| s.parse().ok()).ok_or("bad seed")?,
        };
        Ok(Prepared { it, job })
    }

    /// Run replication `rep`: its statistics, or `None` if the command failed.
    pub fn rep(&mut self, rep: usize) -> Option<Vec<f64>> {
        run_one(&mut self.it, &self.job, rep, false)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn threaded(me: &AdoInterpreter, job: &Job, from: usize, reps: usize, n: usize) -> Vec<Option<Vec<f64>>> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let next = AtomicUsize::new(from);
    let out: Mutex<Vec<Option<Vec<f64>>>> = Mutex::new(vec![None; reps - from]);
    let store = me.engine.store.clone();
    std::thread::scope(|scope| {
        for _ in 0..n {
            let store = store.clone();
            let (next, out) = (&next, &out);
            scope.spawn(move || {
                let mut it = AdoInterpreter::new();
                it.engine.store = store;
                loop {
                    let rep = next.fetch_add(1, Ordering::Relaxed);
                    if rep >= reps {
                        break;
                    }
                    let r = run_one(&mut it, job, rep, false);
                    out.lock().unwrap()[rep - from] = r;
                }
            });
        }
    });
    out.into_inner().unwrap()
}
