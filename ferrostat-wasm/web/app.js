// Ferrostat browser interface. The engine runs in WebAssembly; files live in
// its in-memory file system (upload to add, click to download).
import init, { Ferrostat } from "./pkg/ferrostat_wasm.js";

const $ = (id) => document.getElementById(id);
const PAGE = 200;
const GRAPH_W = 800, GRAPH_H = 600;

let fs;               // the engine
let dataStart = 0;
let selectedVar = null;
let histPos = null;
let shownGraph = -1;

// ---------- results ----------

function appendOutput() {
  const entries = JSON.parse(fs.take_output());
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

async function afterRun() {
  appendOutput();
  $("prompt").textContent = fs.pending() ? ">" : ".";
  renderHistory();
  renderVars();
  if (activeTab() === "data") renderData();
  await rasterJobs();
  renderFiles();
  if (fs.graph_serial() !== shownGraph) {
    shownGraph = fs.graph_serial();
    renderGraph();
    if (fs.graph_svg(GRAPH_W, GRAPH_H)) showTab("graph");
  }
  $("status").textContent = `${fs.n_vars()} vars · ${fs.n_obs()} obs · ${fs.cwd()}`;
}

async function run(line) {
  fs.submit(line);
  await afterRun();
}

async function runScript(text) {
  showTab("results");
  fs.run_script(text);
  await afterRun();
}

// ---------- side panels ----------

function renderHistory() {
  const ul = $("history");
  const hist = JSON.parse(fs.history());
  ul.replaceChildren(...hist.map((h) => {
    const li = document.createElement("li");
    li.textContent = h;
    li.title = "Click to edit, double-click to run";
    li.onclick = () => { $("command").value = h; $("command").focus(); };
    li.ondblclick = () => run(h);
    return li;
  }));
  ul.scrollTop = ul.scrollHeight;
}

function renderVars() {
  const vars = JSON.parse(fs.variables());
  const body = $("vars").tBodies[0];
  body.replaceChildren(...vars.map((v, i) => {
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
  const files = JSON.parse(fs.files());
  const ul = $("files");
  ul.replaceChildren(...files.map((f) => {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = f.path;
    name.title = `${f.path} (${f.size} bytes) — double-click to open`;
    name.ondblclick = () => openFile(f.path);
    const dl = document.createElement("button");
    dl.textContent = "↓";
    dl.title = "Download";
    dl.onclick = () => download(f.path.split("/").pop(), fs.read_file(f.path));
    const rm = document.createElement("button");
    rm.textContent = "✕";
    rm.title = "Delete";
    rm.onclick = () => { fs.remove_file(f.path); renderFiles(); };
    li.append(name, dl, rm);
    return li;
  }));
}

const q = (p) => `"${p}"`;

function openFile(path) {
  const ext = path.split(".").pop().toLowerCase();
  if (ext === "dta") return run(`use ${q(path)}, clear`);
  if (ext === "csv" || ext === "tsv" || ext === "txt") return run(`import delimited using ${q(path)}, clear`);
  if (ext === "xlsx") return run(`import excel using ${q(path)}, firstrow clear`);
  if (ext === "do" || ext === "ado") {
    $("editor").value = new TextDecoder().decode(fs.read_file(path));
    $("editor").dataset.path = path;
    return showTab("editor");
  }
  if (ext === "svg") {
    $("graph").innerHTML = new TextDecoder().decode(fs.read_file(path));
    return showTab("graph");
  }
  download(path.split("/").pop(), fs.read_file(path));
}

async function addFiles(list) {
  for (const f of list) {
    fs.write_file(f.name, new Uint8Array(await f.arrayBuffer()));
  }
  renderFiles();
  const first = list[0];
  if (list.length === 1 && /\.(dta|csv|xlsx|do)$/i.test(first.name)) openFile("/" + first.name);
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

function renderData() {
  const n = fs.n_obs();
  if (dataStart >= n) dataStart = Math.max(0, n - PAGE);
  const vars = JSON.parse(fs.variables());
  const page = JSON.parse(fs.data_page(dataStart, PAGE));
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

function renderGraph() {
  const svg = fs.graph_svg(GRAPH_W, GRAPH_H);
  if (svg) $("graph").innerHTML = svg;
  else $("graph").innerHTML = '<p class="empty">No graph yet.</p>';
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

// `graph export x.png|x.pdf` queues SVG; rasterise it here and store the file.
async function rasterJobs() {
  for (const j of JSON.parse(fs.take_raster_jobs())) {
    const c = await svgToCanvas(j.svg, j.width, j.height);
    if (j.pdf) {
      const rgba = c.getContext("2d").getImageData(0, 0, j.width, j.height).data;
      const rgb = new Uint8Array(j.width * j.height * 3);
      for (let i = 0, k = 0; i < rgba.length; i += 4) { rgb[k++] = rgba[i]; rgb[k++] = rgba[i + 1]; rgb[k++] = rgba[i + 2]; }
      fs.write_file(j.path, Ferrostat.pdf_from_rgb(rgb, j.width, j.height));
    } else {
      const blob = await new Promise((r) => c.toBlob(r, "image/png"));
      fs.write_file(j.path, new Uint8Array(await blob.arrayBuffer()));
    }
  }
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
    run(line);
  };
  $("command").onkeydown = (e) => {
    const hist = JSON.parse(fs.history());
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

  $("btn-upload").onclick = () => $("file-input").click();
  $("file-input").onchange = (e) => { addFiles([...e.target.files]); e.target.value = ""; };
  const right = $("right");
  for (const el of [right, $("center")]) {
    el.addEventListener("dragover", (e) => { e.preventDefault(); right.classList.add("dragover"); });
    el.addEventListener("dragleave", () => right.classList.remove("dragover"));
    el.addEventListener("drop", (e) => { e.preventDefault(); right.classList.remove("dragover"); addFiles([...e.dataTransfer.files]); });
  }

  $("data-prev").onclick = () => { dataStart = Math.max(0, dataStart - PAGE); renderData(); };
  $("data-next").onclick = () => { if (dataStart + PAGE < fs.n_obs()) dataStart += PAGE; renderData(); };

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
    if (name) { fs.write_file(name, new TextEncoder().encode(ed.value)); ed.dataset.path = name; renderFiles(); }
  };

  $("graph-svg").onclick = () => {
    const svg = fs.graph_svg(GRAPH_W, GRAPH_H);
    if (svg) download("graph.svg", new TextEncoder().encode(svg));
  };
  $("graph-png").onclick = async () => {
    const svg = fs.graph_svg(GRAPH_W, GRAPH_H);
    if (!svg) return;
    const c = await svgToCanvas(svg, GRAPH_W, GRAPH_H);
    c.toBlob(async (b) => download("graph.png", new Uint8Array(await b.arrayBuffer())), "image/png");
  };
}

async function main() {
  $("status").textContent = "loading…";
  await init();
  fs = new Ferrostat();
  wire();
  $("results").textContent = `Ferrostat ${Ferrostat.version()} — running in your browser; data never leaves this page.\nUpload .dta, .csv, .xlsx or .do files, or type commands below.\n\n`;
  await afterRun();
  $("command").focus();
}

main().catch((e) => {
  $("status").textContent = "failed to load";
  $("results").textContent = "Could not start Ferrostat: " + e;
});
