// Ferrostat browser interface. The engine runs in a Web Worker (worker.js)
// so long computations never freeze the page; session files are kept in the
// browser's IndexedDB so they survive reloads. Nothing leaves the browser.

const $ = (id) => document.getElementById(id);
const PAGE = 200;
const GRAPH_W = 800, GRAPH_H = 600;

let state = { history: [], variables: [], files: [], nObs: 0, nVars: 0, cwd: "/", pending: false, graphSerial: 0 };
let dataStart = 0;
let selectedVar = null;
let histPos = null;
let shownGraph = 0;

// ---------- persistent file store (IndexedDB) ----------

const store = (() => {
  let dbp = null;
  const open = () => dbp ??= new Promise((resolve, reject) => {
    const req = indexedDB.open("ferrostat", 1);
    req.onupgradeneeded = () => req.result.createObjectStore("files");
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  const tx = async (mode, fn) => {
    const db = await open();
    return new Promise((resolve, reject) => {
      const t = db.transaction("files", mode);
      const out = fn(t.objectStore("files"));
      t.oncomplete = () => resolve(out?.result ?? out);
      t.onerror = () => reject(t.error);
    });
  };
  return {
    async all() {
      try {
        const db = await open();
        return await new Promise((resolve, reject) => {
          const files = [];
          const req = db.transaction("files").objectStore("files").openCursor();
          req.onsuccess = () => {
            const c = req.result;
            if (c) { files.push({ path: c.key, bytes: new Uint8Array(c.value) }); c.continue(); } else resolve(files);
          };
          req.onerror = () => reject(req.error);
        });
      } catch { return []; }
    },
    async apply(changes) {
      if (!changes.length) return;
      try {
        await tx("readwrite", (s) => { for (const c of changes) c.removed ? s.delete(c.path) : s.put(c.bytes, c.path); });
      } catch (e) { console.warn("could not save session files", e); }
    },
    async clear() { try { await tx("readwrite", (s) => s.clear()); } catch { /* storage unavailable */ } },
  };
})();

// ---------- engine (worker RPC) ----------

let worker, nextId = 1;
let restarting = false;
const waiting = new Map();
let queue = Promise.resolve();

function startWorker() {
  worker = new Worker("worker.js", { type: "module" });
  worker.onmessage = (e) => {
    const { id, ok, value, error } = e.data;
    const w = waiting.get(id);
    waiting.delete(id);
    if (w) ok ? w.resolve(value) : w.reject(new Error(error));
  };
  worker.onerror = (e) => console.error("engine error", e.message);
}

function call(op, ...args) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    waiting.set(id, { resolve, reject });
    worker.postMessage({ id, op, args });
  });
}

// commands run one at a time, in the order they were entered
function enqueue(fn) {
  const p = queue.then(fn, fn);
  queue = p.catch(() => {});
  return p;
}

async function restartEngine(reason) {
  restarting = true;
  worker.terminate();
  for (const w of waiting.values()) w.reject(new Error(reason));
  waiting.clear();
  queue = Promise.resolve();
  startWorker();
  await call("restoreFiles", await store.all());
  appendEntries([{ kind: "error", text: `${reason}\nThe engine was restarted: data in memory was cleared; saved files are kept.` }]);
  await applySnapshot(await call("snapshot"));
  restarting = false;
  setBusy(false);
}

// ---------- results ----------

function appendEntries(entries) {
  const res = $("results");
  for (const e of entries) {
    const span = document.createElement("span");
    if (e.kind === "command") span.className = "cmd";
    if (e.kind === "error") span.className = "err";
    span.textContent = e.text + "\n";
    res.appendChild(span);
  }
  res.scrollTop = res.scrollHeight;
}

function setBusy(busy) {
  $("btn-stop").hidden = !busy;
  document.body.classList.toggle("busy", busy);
  $("status").textContent = busy ? "running…" : `${state.nVars} vars · ${state.nObs} obs · ${state.cwd}`;
}

async function applySnapshot(s) {
  appendEntries(s.output);
  state = { ...state, ...s };
  $("prompt").textContent = s.pending ? ">" : ".";
  renderHistory();
  renderVars();
  renderFiles();
  if (activeTab() === "data") renderData();
  await rasterJobs(s.rasterJobs || []);
  if (s.graphSerial !== shownGraph) {
    shownGraph = s.graphSerial;
    if (await renderGraph()) showTab("graph");
  }
  await store.apply(await call("changes"));
}

function run(op, arg) {
  return enqueue(async () => {
    setBusy(true);
    try {
      await applySnapshot(await call(op, arg));
    } catch (e) {
      if (restarting) return; // stopped: restartEngine reports it
      if (/unreachable|RuntimeError|engine/i.test(e.message)) await restartEngine(`Internal error: ${e.message}`);
      else appendEntries([{ kind: "error", text: e.message }]);
    } finally {
      setBusy(false);
    }
  });
}

const submit = (line) => run("submit", line);
const runScript = (text) => { showTab("results"); return run("runScript", text); };

// ---------- side panels ----------

function renderHistory() {
  const ul = $("history");
  ul.replaceChildren(...state.history.map((h) => {
    const li = document.createElement("li");
    li.textContent = h;
    li.title = "Click to edit, double-click to run";
    li.onclick = () => { $("command").value = h; $("command").focus(); };
    li.ondblclick = () => submit(h);
    return li;
  }));
  ul.scrollTop = ul.scrollHeight;
}

function renderVars() {
  const vars = state.variables;
  const body = $("vars").tBodies[0];
  body.replaceChildren(...vars.map((v) => {
    const tr = document.createElement("tr");
    if (v.name === selectedVar) tr.className = "selected";
    for (const t of [v.name, v.label]) {
      const td = document.createElement("td");
      td.textContent = t;
      td.title = t;
      tr.appendChild(td);
    }
    tr.onclick = () => { selectedVar = v.name; renderVars(); };
    tr.ondblclick = () => {
      const c = $("command");
      c.value += (c.value && !c.value.endsWith(" ") ? " " : "") + v.name;
      c.focus();
    };
    return tr;
  }));
  const v = vars.find((x) => x.name === selectedVar);
  const dl = $("props");
  dl.replaceChildren();
  if (v) {
    for (const [k, val] of [["Name", v.name], ["Label", v.label], ["Type", v.type], ["Format", v.format], ["Value label", v.vallab]]) {
      const dt = document.createElement("dt"); dt.textContent = k;
      const dd = document.createElement("dd"); dd.textContent = val;
      dl.append(dt, dd);
    }
  }
}

function renderFiles() {
  const ul = $("files");
  ul.replaceChildren(...state.files.map((f) => {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = f.path;
    name.title = `${f.path} (${f.size} bytes) — double-click to open`;
    name.ondblclick = () => openFile(f.path);
    const dl = document.createElement("button");
    dl.textContent = "↓";
    dl.title = "Download";
    dl.onclick = async () => download(f.path.split("/").pop(), await call("readFile", f.path));
    const rm = document.createElement("button");
    rm.textContent = "✕";
    rm.title = "Delete";
    rm.onclick = () => enqueue(async () => {
      await call("removeFile", f.path);
      await store.apply(await call("changes"));
      state = { ...state, ...(await call("snapshot")) };
      renderFiles();
    });
    li.append(name, dl, rm);
    return li;
  }));
}

const q = (p) => `"${p}"`;

async function openFile(path) {
  const ext = path.split(".").pop().toLowerCase();
  if (ext === "dta") return submit(`use ${q(path)}, clear`);
  if (ext === "csv" || ext === "tsv" || ext === "txt") return submit(`import delimited using ${q(path)}, clear`);
  if (ext === "xlsx") return submit(`import excel using ${q(path)}, firstrow clear`);
  const bytes = await call("readFile", path);
  if (ext === "do" || ext === "ado") {
    $("editor").value = new TextDecoder().decode(bytes);
    $("editor").dataset.path = path;
    return showTab("editor");
  }
  if (ext === "svg") {
    $("graph").innerHTML = new TextDecoder().decode(bytes);
    return showTab("graph");
  }
  download(path.split("/").pop(), bytes);
}

async function addFiles(list) {
  await enqueue(async () => {
    for (const f of list) await call("writeFile", f.name, new Uint8Array(await f.arrayBuffer()));
    await store.apply(await call("changes"));
    state = { ...state, ...(await call("snapshot")) };
    renderFiles();
  });
  const first = list[0];
  const cwd = state.cwd.endsWith("/") ? state.cwd : state.cwd + "/";
  if (list.length === 1 && /\.(dta|csv|xlsx|do)$/i.test(first.name)) openFile(cwd + first.name);
}

function download(name, bytes) {
  if (!bytes) return;
  const url = URL.createObjectURL(new Blob([bytes]));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------- data browser ----------

async function renderData() {
  const n = state.nObs;
  if (dataStart >= n) dataStart = Math.max(0, n - PAGE);
  const vars = state.variables;
  const page = await call("dataPage", dataStart, PAGE);
  const table = $("data-table");
  const head = document.createElement("tr");
  head.append(...["", ...vars.map((v) => v.name)].map((t, i) => {
    const th = document.createElement("th");
    th.textContent = t;
    if (i > 0) th.title = vars[i - 1].label;
    return th;
  }));
  const rows = page.rows.map((r, k) => {
    const tr = document.createElement("tr");
    const num = document.createElement("td");
    num.className = "n";
    num.textContent = dataStart + k + 1;
    tr.appendChild(num);
    r.forEach((cell, c) => {
      const td = document.createElement("td");
      if (page.strings[c]) td.className = "s";
      td.textContent = cell;
      tr.appendChild(td);
    });
    return tr;
  });
  table.replaceChildren(head, ...rows);
  $("data-range").textContent = n ? `${dataStart + 1}–${Math.min(n, dataStart + PAGE)} of ${n}` : "no data";
}

// ---------- graphs ----------

async function renderGraph() {
  const svg = await call("graphSvg", GRAPH_W, GRAPH_H);
  $("graph").innerHTML = svg || '<p class="empty">No graph yet.</p>';
  return !!svg;
}

function svgToCanvas(svg, w, h) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = w; c.height = h;
      const ctx = c.getContext("2d");
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, w, h);
      ctx.drawImage(img, 0, 0, w, h);
      URL.revokeObjectURL(url);
      resolve(c);
    };
    img.onerror = reject;
    img.src = url;
  });
}

// `graph export x.png|x.pdf` arrives as SVG; rasterise it here and store the file.
async function rasterJobs(jobs) {
  for (const j of jobs) {
    const c = await svgToCanvas(j.svg, j.width, j.height);
    let bytes;
    if (j.pdf) {
      const rgba = c.getContext("2d").getImageData(0, 0, j.width, j.height).data;
      const rgb = new Uint8Array(j.width * j.height * 3);
      for (let i = 0, k = 0; i < rgba.length; i += 4) { rgb[k++] = rgba[i]; rgb[k++] = rgba[i + 1]; rgb[k++] = rgba[i + 2]; }
      bytes = await call("pdfFromRgb", rgb, j.width, j.height);
    } else {
      const blob = await new Promise((r) => c.toBlob(r, "image/png"));
      bytes = new Uint8Array(await blob.arrayBuffer());
    }
    await call("writeFile", j.path, bytes);
  }
  if (jobs.length) state = { ...state, ...(await call("snapshot")) };
}

// ---------- tabs ----------

function activeTab() {
  return document.querySelector(".tabs .active").dataset.tab;
}

function showTab(name) {
  for (const b of document.querySelectorAll(".tabs button")) b.classList.toggle("active", b.dataset.tab === name);
  for (const p of document.querySelectorAll(".pane")) p.classList.toggle("active", p.id === "pane-" + name);
  if (name === "data") renderData();
}

// ---------- wiring ----------

function editorSelection() {
  const ed = $("editor");
  return ed.selectionStart !== ed.selectionEnd ? ed.value.slice(ed.selectionStart, ed.selectionEnd) : null;
}

function wire() {
  for (const b of document.querySelectorAll(".tabs button")) b.onclick = () => showTab(b.dataset.tab);

  $("command-form").onsubmit = (e) => {
    e.preventDefault();
    const c = $("command");
    const line = c.value;
    c.value = "";
    histPos = null;
    if (activeTab() !== "results" && activeTab() !== "data") showTab("results");
    submit(line);
  };
  $("command").onkeydown = (e) => {
    const hist = state.history;
    if (e.key === "ArrowUp" && hist.length) {
      histPos = histPos === null ? hist.length - 1 : Math.max(0, histPos - 1);
      e.target.value = hist[histPos];
      e.preventDefault();
    } else if (e.key === "ArrowDown") {
      if (histPos !== null && histPos + 1 < hist.length) { histPos++; e.target.value = hist[histPos]; }
      else { histPos = null; e.target.value = ""; }
      e.preventDefault();
    }
  };

  $("btn-stop").onclick = () => restartEngine("Stopped by the user.");
  $("btn-reset").onclick = async () => {
    if (!confirm("Delete all files saved in this browser for Ferrostat?")) return;
    await store.clear();
    location.reload();
  };
  $("btn-upload").onclick = () => $("file-input").click();
  $("file-input").onchange = (e) => { addFiles([...e.target.files]); e.target.value = ""; };
  const right = $("right");
  for (const el of [right, $("center")]) {
    el.addEventListener("dragover", (e) => { e.preventDefault(); right.classList.add("dragover"); });
    el.addEventListener("dragleave", () => right.classList.remove("dragover"));
    el.addEventListener("drop", (e) => { e.preventDefault(); right.classList.remove("dragover"); addFiles([...e.dataTransfer.files]); });
  }

  $("data-prev").onclick = () => { dataStart = Math.max(0, dataStart - PAGE); renderData(); };
  $("data-next").onclick = () => { if (dataStart + PAGE < state.nObs) dataStart += PAGE; renderData(); };

  const ed = $("editor");
  try { ed.value = localStorage.getItem("ferrostat.dofile") ?? ""; } catch { /* storage unavailable */ }
  ed.oninput = () => { try { localStorage.setItem("ferrostat.dofile", ed.value); } catch { /* ignore */ } };
  ed.onkeydown = (e) => {
    if (e.key === "Tab") {
      e.preventDefault();
      ed.setRangeText("\t", ed.selectionStart, ed.selectionEnd, "end");
    } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      runScript(editorSelection() ?? ed.value);
    }
  };
  $("ed-run").onclick = () => runScript(ed.value);
  $("ed-run-sel").onclick = () => runScript(editorSelection() ?? ed.value);
  $("ed-save").onclick = () => {
    const name = prompt("File name", ed.dataset.path || "analysis.do");
    if (!name) return;
    enqueue(async () => {
      await call("writeFile", name, new TextEncoder().encode(ed.value));
      await store.apply(await call("changes"));
      state = { ...state, ...(await call("snapshot")) };
      ed.dataset.path = name;
      renderFiles();
    });
  };

  $("graph-svg").onclick = async () => {
    const svg = await call("graphSvg", GRAPH_W, GRAPH_H);
    if (svg) download("graph.svg", new TextEncoder().encode(svg));
  };
  $("graph-png").onclick = async () => {
    const svg = await call("graphSvg", GRAPH_W, GRAPH_H);
    if (!svg) return;
    const c = await svgToCanvas(svg, GRAPH_W, GRAPH_H);
    c.toBlob(async (b) => download("graph.png", new Uint8Array(await b.arrayBuffer())), "image/png");
  };
}

async function main() {
  $("status").textContent = "loading…";
  startWorker();
  wire();
  const saved = await store.all();
  await call("restoreFiles", saved);
  $("results").textContent = `Ferrostat ${await call("version")} — running in your browser; data never leaves this page.\n` +
    `Upload .dta, .csv, .xlsx or .do files, or type commands below.` +
    (saved.length ? `\n${saved.length} file(s) restored from this browser's storage.` : "") + "\n\n";
  await applySnapshot(await call("snapshot"));
  setBusy(false);
  $("command").focus();
}

main().catch((e) => {
  $("status").textContent = "failed to load";
  $("results").textContent = "Could not start Ferrostat: " + e;
});
