// Hover readouts for the graph pane. The engine sends a side channel with
// the screen position and values of every marker (one per pixel), line
// vertex and bar; the SVG itself carries no metadata. Hit-testing uses a
// spatial grid so it stays fast for scatters of millions of points.

const CELL = 16;
const RADIUS = 8;
const SVGNS = "http://www.w3.org/2000/svg";

function fmt(v) {
  if (v === null || v === undefined) return ".";
  if (Number.isInteger(v)) return String(v);
  const a = Math.abs(v);
  return a !== 0 && (a < 1e-4 || a >= 1e9) ? v.toExponential(4) : String(+v.toPrecision(7));
}

function buildGrid(map) {
  const cols = Math.ceil(map.w / CELL) + 1, rows = Math.ceil(map.h / CELL) + 1;
  const cells = new Map();
  map.pts.forEach((p, i) => {
    const cx = Math.floor(p[0] / CELL), cy = Math.floor(p[1] / CELL);
    if (cx < 0 || cy < 0 || cx >= cols || cy >= rows) return;
    const k = cy * cols + cx;
    let c = cells.get(k);
    if (!c) cells.set(k, (c = []));
    c.push(i);
  });
  return { cols, rows, cells };
}

function nearestPoint(map, grid, x, y) {
  let best = -1, bd = RADIUS * RADIUS + 1;
  const c0 = Math.floor((x - RADIUS) / CELL), c1 = Math.floor((x + RADIUS) / CELL);
  const r0 = Math.floor((y - RADIUS) / CELL), r1 = Math.floor((y + RADIUS) / CELL);
  for (let cy = Math.max(0, r0); cy <= Math.min(grid.rows - 1, r1); cy++) {
    for (let cx = Math.max(0, c0); cx <= Math.min(grid.cols - 1, c1); cx++) {
      const c = grid.cells.get(cy * grid.cols + cx);
      if (!c) continue;
      for (const i of c) {
        const p = map.pts[i];
        const d = (p[0] - x) ** 2 + (p[1] - y) ** 2;
        // the first recorded point (drawn first) wins ties
        if (d < bd || (d === bd && i < best)) { bd = d; best = i; }
      }
    }
  }
  return best >= 0 ? map.pts[best] : null;
}

function barAt(map, x, y) {
  for (let i = map.bars.length - 1; i >= 0; i--) {
    const b = map.bars[i];
    if (x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]) return b;
  }
  return null;
}

// nearest line vertex in x within the line's panel; several lines: closest vertically
function nearestLinePoint(map, x, y) {
  let best = null, bd = Infinity;
  map.series.forEach((s, k) => {
    const line = s.line;
    const pan = map.panels[s.panel];
    if (!line.length || !pan || x < pan[0] || x > pan[2] || y < pan[1] || y > pan[3]) return;
    let lo = 0, hi = line.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (line[m][0] < x) lo = m + 1; else hi = m; }
    let q = null;
    for (const j of [lo - 1, lo]) {
      const c = line[j];
      if (c && (!q || Math.abs(c[0] - x) < Math.abs(q[0] - x))) q = c;
    }
    if (!q) return;
    // among vertices in the same pixel column, the one nearest vertically
    for (let j = lo; j < line.length && line[j][0] === q[0]; j++) if (Math.abs(line[j][1] - y) < Math.abs(q[1] - y)) q = line[j];
    for (let j = lo - 1; j >= 0 && line[j][0] === q[0]; j--) if (Math.abs(line[j][1] - y) < Math.abs(q[1] - y)) q = line[j];
    const d = Math.abs(q[1] - y) + Math.abs(q[0] - x);
    if (d < bd) { bd = d; best = { p: q, series: k }; }
  });
  return best;
}

function pointText(map, px, py, x, y, obs, series, label) {
  const s = map.series[series] || {};
  let t = `${s.name || "y"} = ${fmt(y)}, ${s.x || "x"} = ${fmt(x)}`;
  if (obs > 0) t += ` (obs ${obs})`;
  if (label) t += ` [${label}]`;
  return { text: t, px, py };
}

function hit(map, grid, x, y) {
  const p = nearestPoint(map, grid, x, y);
  if (p) return pointText(map, p[0], p[1], p[2], p[3], p[4], p[5], p[6]);
  const b = barAt(map, x, y);
  if (b) {
    const s = map.series[b[5]] || {};
    return { text: `${b[6] ? b[6] + ": " : ""}${s.name ? s.name + " = " : ""}${fmt(b[4])}`, px: (b[0] + b[2]) / 2, py: b[1], bar: b };
  }
  const l = nearestLinePoint(map, x, y);
  if (l) return pointText(map, l.p[0], l.p[1], l.p[2], l.p[3], l.p[4], l.series, "");
  return null;
}

let tip = null;
function tooltip() {
  if (!tip) {
    tip = document.createElement("div");
    tip.className = "graph-tip";
    tip.hidden = true;
    document.body.appendChild(tip);
  }
  return tip;
}

// Attach hover readouts to the <svg> inside `box`, using the side channel `json`.
export function attachHover(box, json) {
  const svg = box.querySelector("svg");
  if (tip) tip.hidden = true;
  if (!svg || !json) return;
  let map;
  try { map = JSON.parse(json); } catch { return; }
  if (!map.pts.length && !map.bars.length && !map.series.some((s) => s.line.length)) return;
  const grid = buildGrid(map);
  const ring = document.createElementNS(SVGNS, "circle");
  ring.setAttribute("r", "6");
  ring.setAttribute("fill", "none");
  ring.setAttribute("stroke", "#000");
  ring.setAttribute("stroke-width", "1.5");
  ring.setAttribute("pointer-events", "none");
  ring.style.display = "none";
  svg.appendChild(ring);
  const t = tooltip();
  svg.addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect();
    const sx = map.w / r.width, sy = map.h / r.height;
    const h = hit(map, grid, (e.clientX - r.left) * sx, (e.clientY - r.top) * sy);
    if (!h) { t.hidden = true; ring.style.display = "none"; return; }
    t.textContent = h.text;
    t.hidden = false;
    t.style.left = `${e.clientX + 14}px`;
    t.style.top = `${e.clientY + 14}px`;
    if (h.bar) { ring.style.display = "none"; return; }
    ring.setAttribute("cx", h.px);
    ring.setAttribute("cy", h.py);
    ring.style.display = "";
  });
  svg.addEventListener("mouseleave", () => { t.hidden = true; ring.style.display = "none"; });
}
