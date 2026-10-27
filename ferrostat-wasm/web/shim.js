// Minimal DOM for workers (the engine worker and the helper workers).
// --- minimal DOM for the graph library's text measurement --------------
// Graph labels are laid out by measuring text in a <span>; workers have no
// DOM, so measure with an OffscreenCanvas using the same font settings.
const measure = new OffscreenCanvas(1, 1).getContext("2d");
class FakeElement {
  constructor() { this.text = ""; this.font = "12px sans-serif"; this.size = 12; }
  set textContent(v) { this.text = String(v ?? ""); }
  get textContent() { return this.text; }
  setAttribute(name, value) {
    if (name !== "style") return;
    const get = (k) => (new RegExp(k + "\\s*:\\s*([^;]+)").exec(value) || [])[1];
    const size = parseFloat(get("font-size") || "12");
    this.size = size;
    this.font = `${get("font-style") || "normal"} ${size}px ${get("font-family") || "sans-serif"}`;
  }
  get offsetWidth() { measure.font = this.font; return Math.round(measure.measureText(this.text).width); }
  get offsetHeight() { return Math.round(this.size * 1.15); }
  remove() {}
}
for (const name of ["Window", "HTMLElement"]) {
  self[name] = class { static [Symbol.hasInstance]() { return true; } };
}
self.window = self;
self.document = { body: { append() {}, appendChild() {} }, createElement: () => new FakeElement() };
