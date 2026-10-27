// Helper worker for parallel replications (see parallel.js): its own engine
// instance runs replications of the job it is sent.
import "./shim.js";
import init, { ParallelJob } from "./pkg/ferrostat_wasm.js";

const ready = init().then(() => postMessage("ready"));

self.onmessage = async (e) => {
  const { jobText, data, from, reps, k } = e.data;
  const ctrl = new Int32Array(e.data.ctrl);
  const out = new Float64Array(e.data.out);
  let job = null;
  try {
    await ready;
    job = new ParallelJob(jobText, data);
    for (;;) {
      const rep = Atomics.add(ctrl, 0, 1);
      if (rep >= reps) break;
      let v;
      try { v = job.rep(rep); } catch { v = undefined; }
      const at = (rep - from) * (k + 1);
      if (v && v.length === k) {
        out.set(v, at + 1);
        out[at] = 1;
      }
    }
  } catch {
    // could not set up the job: take no replications
  } finally {
    if (job) job.free();
    Atomics.add(ctrl, 1, 1);
    Atomics.notify(ctrl, 1);
  }
};
