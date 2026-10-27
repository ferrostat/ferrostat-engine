// Runs the Ferrostat engine off the page's main thread. The page sends
// {id, op, args} messages and receives {id, ok, value | error}.
import "./shim.js";
import "./parallel.js";
import init, { Ferrostat } from "./pkg/ferrostat_wasm.js";

// --- engine ----------------------------------------------------------------
let fs = null;
const ready = init().then(() => { fs = new Ferrostat(); });

// state the page needs after every command
function snapshot() {
  return {
    output: JSON.parse(fs.take_output()),
    history: JSON.parse(fs.history()),
    variables: JSON.parse(fs.variables()),
    files: JSON.parse(fs.files()),
    nObs: fs.n_obs(),
    nVars: fs.n_vars(),
    cwd: fs.cwd(),
    pending: fs.pending(),
    graphSerial: fs.graph_serial(),
    graphVisible: fs.graph_visible(),
    rasterJobs: JSON.parse(fs.take_raster_jobs()),
  };
}

// changed files with their contents, for persistence
function changes() {
  return JSON.parse(fs.take_changes()).map((c) => (c.removed ? c : { ...c, bytes: fs.read_file(c.path) }));
}

const ops = {
  version: () => Ferrostat.version(),
  commandWords: () => JSON.parse(fs.command_words()),
  submit: (line) => { fs.submit(line); return snapshot(); },
  runScript: (text) => { fs.run_script(text); return snapshot(); },
  snapshot: () => snapshot(),
  changes: () => changes(),
  dataPage: (start, count) => JSON.parse(fs.data_page(start, count)),
  graphSvg: (w, h) => fs.graph_svg(w, h) ?? null,
  graphHover: () => fs.graph_hover(),
  writeFile: (path, bytes) => { fs.write_file(path, bytes); },
  // restoring saved files must not report them as changed again
  restoreFiles: (files) => { for (const f of files) fs.write_file(f.path, f.bytes); fs.take_changes(); },
  readFile: (path) => fs.read_file(path) ?? null,
  removeFile: (path) => fs.remove_file(path),
  pdfFromRgb: (rgb, w, h) => Ferrostat.pdf_from_rgb(rgb, w, h),
};

self.onmessage = async (e) => {
  const { id, op, args } = e.data;
  try {
    await ready;
    const value = ops[op](...(args || []));
    const transfer = value instanceof Uint8Array ? [value.buffer] : [];
    self.postMessage({ id, ok: true, value }, transfer);
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err && err.message || err) });
  }
};
