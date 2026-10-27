// Parallel replications for bootstrap/simulate/permute in the browser.
//
// The engine calls ferrostatParallel(job, data) synchronously in the middle
// of a command. Helper workers (helper.js), each with its own engine
// instance, take replications from a shared counter and write their
// statistics into shared memory while this worker waits on Atomics.wait.
// Shared memory needs a cross-origin isolated page (coi-sw.js); without it,
// or before the helpers are ready, this returns undefined and the engine
// runs the replications itself, with the same random streams.

const helpers = [];

function startHelpers() {
  if (!self.crossOriginIsolated || typeof SharedArrayBuffer === "undefined") return;
  // the engine worker only waits while helpers run, so one helper per core
  const n = Math.min(navigator.hardwareConcurrency || 1, 16);
  for (let i = 0; i < n; i++) {
    const w = new Worker(new URL("./helper.js", import.meta.url), { type: "module" });
    const h = { worker: w, ready: false };
    w.onmessage = (e) => { if (e.data === "ready") h.ready = true; };
    helpers.push(h);
  }
}

self.ferrostatParallel = (jobText, data) => {
  const { from, reps, k, workers } = JSON.parse(jobText);
  const ready = helpers.filter((h) => h.ready).slice(0, Math.max(workers || 1, 1));
  if (ready.length === 0) return undefined;
  const n = reps - from;
  // ctrl[0]: next replication to take; ctrl[1]: helpers finished
  const ctrl = new Int32Array(new SharedArrayBuffer(8));
  ctrl[0] = from;
  // per replication: a flag (1 = ran) and its k statistics
  const out = new Float64Array(new SharedArrayBuffer(8 * Math.max(n * (k + 1), 1)));
  const bytes = data.slice();
  for (const h of ready) h.worker.postMessage({ jobText, data: bytes, ctrl: ctrl.buffer, out: out.buffer, from, reps, k });
  let done;
  while ((done = Atomics.load(ctrl, 1)) < ready.length) Atomics.wait(ctrl, 1, done);
  // a helper that could not set up the job takes no replications: if some
  // were never taken, let the engine run the job itself
  if (Atomics.load(ctrl, 0) < reps) return undefined;
  const res = new Array(n);
  for (let i = 0; i < n; i++) {
    const at = i * (k + 1);
    res[i] = out[at] === 1 ? Array.from(out.subarray(at + 1, at + 1 + k)) : null;
  }
  return JSON.stringify(res);
};

startHelpers();
