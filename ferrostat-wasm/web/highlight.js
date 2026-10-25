// Syntax highlighting for do-files. Pure function: text in, HTML out.
//
// Commands are coloured only where a command can appear: at the start of a
// statement, after a prefix (quietly, capture, by ...:), after `{`, or after
// `;` under `#delimit ;`. The command list comes from the engine itself.

const KEYWORDS = new Set([
  "foreach", "forvalues", "while", "if", "else", "in", "of", "local", "global", "program", "end",
  "capture", "quietly", "noisily", "by", "bysort", "continue", "break", "exit", "version", "do", "run", "include",
  "preserve", "restore", "return", "ereturn", "syntax", "args", "tempvar", "tempname", "tempfile",
]);
// prefixes after which the next word is again a command
const PREFIXES = new Set(["capture", "cap", "quietly", "qui", "noisily", "noi", "xi", "version", "nobreak", "timer"]);

const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const span = (cls, s) => `<span class="tk-${cls}">${esc(s)}</span>`;
// strings are expanded at run time: show the macros inside them
const stringSpan = (s) => `<span class="tk-string">${s.split(/(`[^`'"]*'|\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*)/).map((part, k) => (k % 2 ? span("macro", part) : esc(part))).join("")}</span>`;

/** Highlight a whole do-file. `commands` is a Set of command words. */
export function highlight(text, commands) {
  let out = "";
  let depth = 0; // nesting of /* */ comments
  let delimitSemicolon = false;
  let continuation = false; // previous line ended with ///
  for (const line of text.split("\n")) {
    out += highlightLine(line) + "\n";
  }
  return out;

  function highlightLine(line) {
    let html = "";
    let i = 0;
    const continued = continuation;
    let atStart = !continued; // a command may appear here
    continuation = false;
    const n = line.length;

    // inside a block comment carried over from previous lines
    if (depth > 0) {
      const j = scanBlock(line, 0);
      html += span("comment", line.slice(0, j));
      i = j;
    }
    // whole-line comment: optional whitespace then *
    if (depth === 0 && i === 0) {
      const m = /^(\s*)\*/.exec(line);
      if (m && !continued) return esc(m[1]) + span("comment", line.slice(m[1].length));
    }
    // #delimit
    const dm = /^\s*#delimit\s*(\S*)/.exec(line);
    if (dm && i === 0) {
      delimitSemicolon = dm[1] === ";";
      return span("keyword", line);
    }

    while (i < n) {
      const c = line[i];
      const rest = line.slice(i);
      // block comment
      if (rest.startsWith("/*")) {
        depth = 1;
        const j = scanBlock(line, i + 2);
        html += span("comment", line.slice(i, j));
        i = j;
        continue;
      }
      // line comment (//, ///) when at the start or after whitespace
      if (rest.startsWith("//") && (i === 0 || /\s/.test(line[i - 1]))) {
        if (rest.startsWith("///")) continuation = true;
        html += span("comment", rest);
        break;
      }
      // compound string `"...."'
      if (rest.startsWith('`"')) {
        const j = line.indexOf("\"'", i + 2);
        const end = j < 0 ? n : j + 2;
        html += stringSpan(line.slice(i, end));
        i = end;
        atStart = false;
        continue;
      }
      // local macro `name' (may nest)
      if (c === "`") {
        let d = 0, j = i;
        for (; j < n; j++) {
          if (line[j] === "`") d++;
          else if (line[j] === "'" && --d === 0) break;
        }
        const end = Math.min(n, j + 1);
        html += span("macro", line.slice(i, end));
        i = end;
        atStart = false;
        continue;
      }
      // global macro $name or ${name}
      if (c === "$") {
        const m = /^\$(\{[^}]*\}|[A-Za-z_][A-Za-z0-9_]*)/.exec(rest);
        if (m) {
          html += span("macro", m[0]);
          i += m[0].length;
          atStart = false;
          continue;
        }
      }
      // string
      if (c === '"') {
        let j = i + 1;
        while (j < n && line[j] !== '"') j++;
        const end = Math.min(n, j + 1);
        html += stringSpan(line.slice(i, end));
        i = end;
        atStart = false;
        continue;
      }
      // number or missing value
      const num = /^(\d+\.?\d*(e[+-]?\d+)?|\.\d+(e[+-]?\d+)?|\.[a-z]?(?![A-Za-z0-9_]))/i.exec(rest);
      if (num && (i === 0 || !/[A-Za-z0-9_.]/.test(line[i - 1]))) {
        html += span("number", num[0]);
        i += num[0].length;
        atStart = false;
        continue;
      }
      // identifier
      const id = /^[A-Za-z_][A-Za-z0-9_]*/.exec(rest);
      if (id) {
        const w = id[0];
        const after = line.slice(i + w.length);
        let cls = null;
        if (atStart && (commands.has(w) || KEYWORDS.has(w))) cls = KEYWORDS.has(w) ? "keyword" : "command";
        else if (KEYWORDS.has(w) && ["in", "of", "if", "else"].includes(w)) cls = "keyword";
        else if (/^\(/.test(after)) cls = "function";
        html += cls ? span(cls, w) : esc(w);
        i += w.length;
        // after a prefix the next word is a command again; `by ...:` handled at ':'
        atStart = atStart && PREFIXES.has(w);
        continue;
      }
      // punctuation
      if (c === ":" || c === "{" || (c === ";" && delimitSemicolon)) atStart = true;
      else if (!/\s/.test(c)) atStart = false;
      html += esc(c);
      i++;
    }
    return html;
  }

  // scan a /* */ comment starting inside it at `from`; returns end index
  function scanBlock(line, from) {
    let j = from;
    while (j < line.length && depth > 0) {
      if (line.startsWith("/*", j)) { depth++; j += 2; }
      else if (line.startsWith("*/", j)) { depth--; j += 2; }
      else j++;
    }
    return j;
  }
}
